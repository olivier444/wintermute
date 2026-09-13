from __future__ import annotations

from typing import Callable, List, Optional, Dict, Any, cast

import time

import torch
import torch.nn as nn

import json
from wintermute.data.iterate.dataclasses import TrainingExample, TrainingViewConfig
from wintermute.data.iterate.training_view import AbstractTrainingView
from wintermute.ml.runs.checkpoints import CheckPoint, RunCheckpoints
from wintermute.tools.model import (
    clear_cuda_memory,
    count_parameters,
    iter_batches,
    move_optimizer_state,
    unload_from_gpu,
)
from wintermute.tools.misc import utc_now
from wintermute.ml.training.config import TrainingConfig
from wintermute.ml.training.benchmark_runner import TrainingBenchmarkRunner
from wintermute.ml.training.eval_runner import EvaluationRunner
from wintermute.ml.training.eval_suite import RuntimeEvalSuite
from wintermute.ml.training.logger import Logger, LogLevel
from wintermute.ml.training.metrics import TrainingMetricsLogger
from wintermute.ml.training.optimizer_diagnostics import OptimizerDiagnosticsTracker
from wintermute.ml.training.parameter_updater import TrainingParameterUpdater
from wintermute.ml.training.runtime import BurnContext, MacroStepContext
from wintermute.ml.training.scaling_law import ScalingLawTracker
from wintermute.ml.training.controller import TrainerController, TrainerCommand, slack_command_help_text
from wintermute.ml.tasks.factory import TaskWrapper
from wintermute.ml.training.lr_schedule import LrSchedule
from wintermute.ml.models.base import BaseModel, count_optimizer_param_groups
from wintermute.tools.logging import console_log


class Trainer:
    _BURN_LOG_INTERVAL_SEC = 600.0

    def __init__(
        self,
        output_root: str,
        run_id: str,
        run_controller: TrainerController,
        task_wrapper: TaskWrapper,
        eval_suites: List[RuntimeEvalSuite],
        model: BaseModel,
        config: TrainingConfig,
        checkpoints: RunCheckpoints,
        logger: Logger,
        device: torch.device,
        save_model: Callable[[nn.Module, str], bool],
        resume_checkpoint: str = "last",
        initial_optimizer_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.output_root = output_root
        self.run_controller = run_controller
        self.model = model
        self.config = config
        self.checkpoints = checkpoints
        self.logger = logger
        self.device = device
        self.save_model = save_model
        self.resume_checkpoint = resume_checkpoint
        self.model.to(self.device)
        self.model.train()
        trainable_param_count = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        self.metrics_logger = TrainingMetricsLogger(
            self.logger,
            trainable_param_count=trainable_param_count,
        )
        self.evaluation_runner = EvaluationRunner(
            output_root=self.output_root,
            model=self.model,
            metrics_logger=self.metrics_logger,
            suites=eval_suites,
        )
        self.benchmark_runner = (
            TrainingBenchmarkRunner(
                output_root=self.output_root,
                model=self.model,
                task_wrapper=task_wrapper,
                checkpoints=self.checkpoints,
                config=self.config.benchmark_config,
                metrics_logger=self.metrics_logger,
                logger=self.logger,
            )
            if self.config.benchmark_config is not None
            else None
        )

        self.lr_schedule = LrSchedule(self.config.lr_config)

        optimizer_param_groups = self.model.build_optimizer_param_groups()
        
        decay_param_count, no_decay_param_count = count_optimizer_param_groups(
            optimizer_param_groups
        )
        console_log(
            "optimizer",
            f"adamw param groups: decay={decay_param_count}, no_decay={no_decay_param_count}, weight_decay={self.config.optimizer_config.weight_decay}",
        )

        self.optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=self.config.lr_config.start_lr,
            fused=True,
            weight_decay=0.0,
        )

        if initial_optimizer_state is not None:
            self.optimizer.load_state_dict(initial_optimizer_state)
            move_optimizer_state(self.optimizer, self.device)
            console_log("optimizer", "loaded reference optimizer state")

        self.task_wrapper = task_wrapper
        self.training_monitor = self.task_wrapper.build_training_monitor(
            model=self.model,
            logger=self.logger,
            configs=self.config.probe_configs,
        )
        self.optimizer_diagnostics = OptimizerDiagnosticsTracker(
            model=self.model,
            optimizer=self.optimizer,
            task_wrapper=self.task_wrapper,
            gradient_noise_update_batch_count=(
                self.config.trainer_config.gradient_noise_update_batch_count
            ),
            gradient_noise_modulus=(
                self.config.trainer_config.gradient_noise_modulus
            ),
        )
        self.config.trainer_config.gradient_noise_modulus = (
            self.optimizer_diagnostics.gradient_noise_modulus
        )
        self.config.trainer_config.gradient_noise_update_batch_count = (
            self.optimizer_diagnostics.gradient_noise_update_batch_count
        )
        self.parameter_updater = TrainingParameterUpdater(
            config=self.config,
            optimizer=self.optimizer,
            lr_schedule=self.lr_schedule,
            optimizer_diagnostics=self.optimizer_diagnostics,
            evaluation_runner=self.evaluation_runner,
            benchmark_runner=self.benchmark_runner,
        )
        self.parameter_updater.apply_optimizer_config()
        self._sync_runtime_parameters()

        baseline_config = dict(config.baseline_config)
        baseline_config.setdefault("batch_size", config.trainer_config.batch_size)
        self.baseline_suite = self.task_wrapper.build_baseline_suite(baseline_config)
        self.global_step = 0
        self._units_at_last_weight_update = 0
        self._paused = False
        self._training_state_parked = False
        self._pending_model_save_pcts = list(self.config.trainer_config.model_save_pcts)
        self.scaling_law_tracker = ScalingLawTracker()

    def train(self) -> nn.Module:
        self.logger.start_fit()
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._log("train", f"compiling model")
        self.model.compile()
        self._log("train", f"model-size: {count_parameters(self.model)} params")

        self.logger.log_params_text(self.config.to_dict())

        self._log("dataset", f"training budget: {self.max_training_units} units")
        self._update_params()
        resumed_cp = self._try_resume_from_checkpoint()

        self.global_step = resumed_cp.global_step if resumed_cp is not None else 0
        units_seen = 0
        examples_seen = 0
        last_train_loss = float("nan")
        macro_step_context = MacroStepContext()

        self.optimizer.zero_grad(set_to_none=True)

        training_view = self._build_data_view(self.config.training_view_config)

        self._compute_baseline_metrics()
        if resumed_cp is None:
            self._log_initial_model_metrics()

        burn_context: Optional[BurnContext] = None

        stop_requested = False
        training_limit_reached = False
        epoch = 0
        while True:

            self._log("train", f"starting training pass {epoch} - units seen={units_seen}")
            units_at_pass_start = units_seen
            for batch in iter_batches(training_view, self.config.trainer_config.batch_size):
                if self._check_controller_and_stop():
                    stop_requested = True
                    break

                if units_seen >= self.max_training_units:
                    training_limit_reached = True
                    break

                batch_units = self._count_batch_units(batch)

                in_burn = resumed_cp is not None and units_seen < resumed_cp.units_seen
                if in_burn:
                    if burn_context is None:
                        burn_context = BurnContext()
                else:
                    if burn_context is not None and resumed_cp is not None:
                        self._log_burn_completion(
                            channel="train",
                            label="resume catch-up",
                            progress=burn_context,
                            units_seen=units_seen,
                            examples_seen=examples_seen,
                            target_units=resumed_cp.units_seen,
                        )
                    burn_context = None
                    capture_observed_loss = (
                        batch_units > 0
                        and self.optimizer_diagnostics.should_capture_observed_loss()
                    )
                    cpu_rng_state = torch.get_rng_state() if capture_observed_loss else None
                    cuda_rng_state = torch.cuda.get_rng_state_all() if capture_observed_loss and torch.cuda.is_available() else None

                    loss = self.task_wrapper.compute_loss(self.model, batch)
                    batch_weight_in_gradient = batch_units / self._grad_accum_size_units
                    (loss * batch_weight_in_gradient).backward()
                    last_train_loss = float(loss.detach().item())
                    macro_step_context.on_micro_batch(last_train_loss, batch_units)
                    self.optimizer_diagnostics.record_micro_batch(
                        batch=batch,
                        batch_units=batch_units,
                        grad_accum_size_units=self._grad_accum_size_units,
                        loss_before_update=last_train_loss,
                        cpu_rng_state=cpu_rng_state,
                        cuda_rng_state=cuda_rng_state,
                    )

                units_seen += batch_units
                examples_seen += len(batch)

                if in_burn and resumed_cp is not None and burn_context is not None:
                    self._log_burn_progress(
                        channel="train",
                        label="resume catch-up",
                        progress=burn_context,
                        units_seen=units_seen,
                        examples_seen=examples_seen,
                        target_units=resumed_cp.units_seen,
                    )
                    self._units_at_last_weight_update = units_seen
                    continue

                if batch_units == 0:
                    continue

                if (units_seen - self._units_at_last_weight_update) >= self._grad_accum_size_units:
                    self._optimizer_step(units_seen)

                    self.metrics_logger.log_update(
                        global_step=self.global_step,
                        epoch=epoch,
                        units_seen=units_seen,
                        examples_seen=examples_seen,
                        last_train_loss=last_train_loss,
                        total_units=self.lr_schedule.total_units,
                        learning_rate=self.lr_schedule.current_lr,
                        grad_accum_size_units=self._grad_accum_size_units,
                        diagnostics=self.optimizer_diagnostics.summary(),
                    )

                    if self.global_step % self._checkpoint_every == 0:
                        self._on_macro_step(
                            epoch,
                            units_seen=units_seen,
                            examples_seen=examples_seen,
                            macro_step_context=macro_step_context
                        )
                        self._log_training_view_iteration_stats(
                            training_view,
                            context=f"train_live_epoch_{epoch}",
                            log_text=False,
                        )
                        macro_step_context.reset()
                        self.optimizer_diagnostics.reset_macro_step()

            self._log_training_view_iteration_stats(training_view, context=f"train_epoch_{epoch}", log_text=True)
            if stop_requested or training_limit_reached:
                break
            if units_seen == units_at_pass_start:
                raise RuntimeError("Training view produced no supervised units.")
            epoch += 1

        if (units_seen - self._units_at_last_weight_update) >= 0.5 * self._grad_accum_size_units: # 50% tolerance on last step
            self._optimizer_step(units_seen)

            self._on_macro_step(
                epoch,
                units_seen=units_seen,
                examples_seen=examples_seen,
                macro_step_context=macro_step_context
            )      

        if stop_requested:
            self._log("train", "training completed after manual stop request", level=LogLevel.WARN)
            return self.model

        if training_limit_reached:
            self._log(
                "train",
                f"training budget reached: {units_seen} / {self.max_training_units} units",
            )

        self._log("train", f"training completed")
        return self.model

    def _compute_baseline_metrics(self) -> None:
        try:
            self.model = unload_from_gpu(self.model)

            self._log("train", f"computing eval metrics")
            for eval_suite in self.evaluation_runner.suites:
                baseline_metrics = self.baseline_suite.compute_eval_metrics(eval_suite.view_config)
                self._log("train", f"baseline metrics computed for {eval_suite.name}")
                self.logger.log_params_text(
                    baseline_metrics,
                    f"baselines/metrics/{eval_suite.name}",
                )
            if self.benchmark_runner is not None:
                baseline_benchmarks = self.benchmark_runner.run_baselines(
                    self.baseline_suite
                )
                for task_name, metrics in baseline_benchmarks.items():
                    self.logger.log_params_text(
                        metrics,
                        f"baselines/benchmarks/{task_name}",
                    )
        finally:
            self.model = self.model.to(self.device)


    def _check_controller_and_stop(self) -> bool:
        stop_requested = self._apply_commands(self.run_controller.poll())
        announced_pause = False

        while self._paused and not stop_requested:
            if not announced_pause:
                self._log("train", "manual pause request detected")
                self._park_training_state_in_cpu_memory()
                self._log("train", "training state parked in CPU RAM and CUDA cache cleared")
                announced_pause = True

            time.sleep(5.0)
            stop_requested = self._apply_commands(self.run_controller.poll())

        if announced_pause and not stop_requested:
            self._restore_training_state_to_device()
            self._log("train", "training resumed")

        return stop_requested
       

    def _update_params(self, params: Optional[Dict[str, Any]] = None) -> None:
        messages = self.parameter_updater.apply(params or {})
        self._sync_runtime_parameters()
        for message in messages:
            self._log("train", message)

    def _sync_runtime_parameters(self) -> None:
        trainer_config = self.config.trainer_config
        self.max_training_units = trainer_config.max_training_units
        self._checkpoint_every = max(1, int(trainer_config.checkpoint_every_batch))
        self._grad_accum_size_units = max(
            1,
            int(trainer_config.grad_accum_size_units),
        )


    def _on_macro_step(
            self,
            epoch: int,
            units_seen: int,
            examples_seen: int,
            macro_step_context: MacroStepContext):

        self.optimizer_diagnostics.estimate_gradient_noise_if_due()
        
        macro_step_context.on_eval_started()
        eval_result = self.evaluation_runner.run_due(global_step=self.global_step)
        macro_step_context.on_eval_completed()

        scaling_fit = self.scaling_law_tracker.record(
            cumulative_tokens=units_seen,
            eval_loss=eval_result.primary_loss,
        )
        if scaling_fit is not None:
            self.metrics_logger.log_scaling_law(
                fit=scaling_fit,
                global_step=self.global_step,
            )

        checkpoint = self._register_checkpoint(
            epoch=epoch,
            units_seen=units_seen,
            examples_seen=examples_seen,
            train_loss=macro_step_context.compute_macro_train_loss(),
            val_loss=eval_result.primary_loss,
        )
        if self.benchmark_runner is not None:
            self.benchmark_runner.run_after_checkpoint(checkpoint)
        self._save_intermediate_models_if_needed(units_seen)

        self._run_due_monitor(macro_step_context)

        self.metrics_logger.log_macro_step(
            global_step=self.global_step,
            units_seen=units_seen,
            context=macro_step_context,
            diagnostics=self.optimizer_diagnostics.summary(),
        )

        macro_step_context.on_macro_step_completed(units_seen=units_seen, global_step=self.global_step)


    def _run_due_monitor(
        self,
        macro_step_context: MacroStepContext | None = None,
    ) -> None:
        if not self.training_monitor.is_due():
            return

        if macro_step_context is not None:
            macro_step_context.on_monitor_started()
        try:
            self.training_monitor.run_due(global_step=self.global_step)
        finally:
            if macro_step_context is not None:
                macro_step_context.on_monitor_completed()

    def _log_initial_model_metrics(self) -> None:
        self._log("train", "computing initial model metrics at step 0")
        self.evaluation_runner.run_all(global_step=self.global_step)
        self.training_monitor.run_initial(global_step=self.global_step)
        

    def _optimizer_step(self, units_seen: int) -> None:
        self._units_at_last_weight_update = units_seen
        self.global_step += 1
       
        self._update_lr(units_seen)

        pending_snapshot = self.optimizer_diagnostics.prepare_optimizer_step()
        raw_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer_config.grad_clip,
            ).detach()
        )
        
        self.optimizer.step()
        self.optimizer_diagnostics.complete_optimizer_step(
            pending_snapshot,
            raw_grad_norm=raw_grad_norm,
            grad_clip=self.config.optimizer_config.grad_clip,
        )

        self.optimizer.zero_grad(set_to_none=True)

    
    def _update_lr(self, units_seen: int):
        self.lr_schedule.update_lr(units_seen)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule.current_lr


    def _register_checkpoint(
        self,
        epoch: int,
        units_seen: int,
        examples_seen: int,
        train_loss: float,
        val_loss: float,
    ) -> CheckPoint:
        checkpoint = CheckPoint(
            checkpoint_id=f"step_{epoch}_{self.global_step}",
            states={
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            global_step=self.global_step,
            units_seen=units_seen,
            examples_seen=examples_seen,
            created_at=utc_now(),
            train_loss=train_loss,
            val_loss=val_loss,
        )
        self.checkpoints.register(checkpoint)
        return checkpoint


    def _save_intermediate_models_if_needed(self, units_seen: int) -> None:
        if len(self._pending_model_save_pcts) == 0:
            return

        total_units = max(1, self.lr_schedule.total_units)
        progress_pct = (100.0 * units_seen) / total_units
        while self._pending_model_save_pcts and progress_pct >= self._pending_model_save_pcts[0]:
            pct = self._pending_model_save_pcts.pop(0)
            dirname = self._intermediate_model_dirname(pct)
            if not self.save_model(self.model, dirname):
                self._log("train", f"intermediate model '{dirname}' already exists; skipping")
                continue

            self._log(
                "train",
                f"saved intermediate model '{dirname}' at progress={progress_pct:.2f}% (target={pct:.2f}%)",
            )


    def _try_resume_from_checkpoint(self) -> Optional[CheckPoint]:
        checkpoint = self.checkpoints.selected(self.resume_checkpoint, with_states=True)
        if checkpoint is None:
            self._log(
                "train",
                f"no checkpoint named '{self.resume_checkpoint}' found for {self.run_id}",
            )
            return None

        model_state = checkpoint.states.get("model")
        optimizer_state = checkpoint.states.get("optimizer")
        if model_state is None or optimizer_state is None:            
            self._log(
                "train",
                f"checkpoint '{checkpoint.checkpoint_id}' found but missing model/optimizer states.",
                level=LogLevel.WARN,
            )
            return None

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        move_optimizer_state(self.optimizer, self.device)
        self.model.to(self.device)
        self.model.train()

        msg = (
            f"RESUME ENABLED: loaded checkpoint '{checkpoint.checkpoint_id}' "
            f"(selector='{self.resume_checkpoint}') for run_id='{self.run_id}' "
            f"(step={checkpoint.global_step}, units_seen={checkpoint.units_seen}, "
            f"examples_seen={checkpoint.examples_seen}, train_loss={checkpoint.train_loss:.6f})."
        )
        self._log("train", msg)

        return checkpoint


    def _cuda_memory_summary(self) -> str:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return "CUDA unavailable"

        alloc_mb = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(self.device) / 1024**2
        return f"cuda_allocated_mb={alloc_mb:.2f}, cuda_reserved_mb={reserved_mb:.2f}"


    def _park_training_state_in_cpu_memory(self) -> None:
        if self._training_state_parked:
            self._log("train", "training state already parked in CPU RAM")
            return

        self._log("train", f"parking training state to CPU RAM ({self._cuda_memory_summary()})")
        move_optimizer_state(self.optimizer, "cpu")
        self.model = unload_from_gpu(self.model)
        clear_cuda_memory()
        self._training_state_parked = True
        self._log("train", f"training state parked in CPU RAM ({self._cuda_memory_summary()})")


    def _restore_training_state_to_device(self) -> None:
        if not self._training_state_parked:
            self._log("train", "training state already active on training device")
            return

        self._log("train", f"restoring training state to {self.device} ({self._cuda_memory_summary()})")
        self.model = self.model.to(self.device)
        move_optimizer_state(self.optimizer, self.device)
        self.model.train()
        self._training_state_parked = False
        self._log("train", f"training state restored to {self.device} ({self._cuda_memory_summary()})")


    def _build_data_view(
        self,
        cfg: TrainingViewConfig,
        silent: bool = False,
    ) -> AbstractTrainingView:
        view = self.task_wrapper.build_training_view(
            output_root=self.output_root,
            training_view_config=cfg,
            silent=silent,
        )

        return cast(AbstractTrainingView, view)


    def _log_training_view_iteration_stats(
        self,
        view: AbstractTrainingView,
        *,
        context: str,
        log_text: bool,
    ) -> None:
        stats = view.last_iteration_stats
        if not stats.has_sft_activity():
            return

        if log_text:
            self._log("dataset", f"{context} sft stats: {stats.describe()}")
        self.metrics_logger.log_training_view(stats, global_step=self.global_step)


    def _intermediate_model_dirname(self, pct: float) -> str:
        pct_label = f"{pct:.6f}".rstrip("0").rstrip(".").replace(".", "_")
        return f"model_{pct_label}pct"
    

    def _count_batch_units(self, batch: List[TrainingExample]) -> int:
        return sum(example.effective_units() for example in batch)


    def _apply_commands(self, commands: List[TrainerCommand]) -> bool:
        stop_requested = False
        for command in commands:
            if command.kind == "help":
                self.logger.log_text("help", slack_command_help_text(), step=self.global_step)
            elif command.kind == "update_params":
                params = command.params or {}
                self._log("train", f"parameter update detected:\n{json.dumps(params, ensure_ascii=True, indent=2)}")
                self._update_params(params)
            elif command.kind == "pause":
                self._paused = True
            elif command.kind == "resume":
                self._paused = False
            elif command.kind == "log":
                if len(self.training_monitor.previews) == 0:
                    self.logger.log_text("log", "no preview available yet", step=self.global_step)
                    continue

                self.logger.publish_preview_snapshot(
                    self.training_monitor.previews
                )
            elif command.kind == "metrics":
                self.logger.publish_metrics_snapshot(command.metrics_offset)
            elif command.kind == "checkpoint":
                self._clone_last_checkpoint(command.checkpoint_name)
            elif command.kind == "stop":
                self._log("train", "manual stop request detected", level=LogLevel.WARN)
                stop_requested = True

        return stop_requested

    def _clone_last_checkpoint(self, name: Optional[str]) -> None:
        if name is None:
            self._log("checkpoint", "checkpoint command is missing a name", level=LogLevel.WARN)
            return

        try:
            checkpoint = self.checkpoints.clone_last(name)
        except Exception as exc:
            self._log("checkpoint", f"named checkpoint '{name}' was not created: {exc}", level=LogLevel.WARN)
            return

        self._log(
            "checkpoint",
            f"cloned checkpoint '{checkpoint.checkpoint_id}' as '{name}' "
            f"(step={checkpoint.global_step}, train_loss={checkpoint.train_loss:.6f}, "
            f"val_loss={checkpoint.val_loss:.6f})",
        )


    def _log(self, channel: str, msg: str, level: LogLevel = LogLevel.INFO) -> None:
        console_log("trainer", f"[{channel}] {msg}")
        self.logger.log_text(channel, msg, step=self.global_step, level=level)


    def _log_burn_progress(
        self,
        *,
        channel: str,
        label: str,
        progress: BurnContext,
        units_seen: int,
        examples_seen: int,
        target_units: Optional[int] = None,
    ) -> None:
        now = time.perf_counter()
        if now - progress.last_log_at < self._BURN_LOG_INTERVAL_SEC:
            return

        elapsed_sec = now - progress.started_at
        units_per_sec = units_seen / elapsed_sec if elapsed_sec > 0 else 0.0

        msg = (
            f"{label} in progress: "
            f"elapsed_sec={elapsed_sec:.1f}, "                
            f"examples_seen={examples_seen}, "
            f"units_seen={units_seen}, "  
            f"units_per_sec={units_per_sec:.1f}"                      
        )        
        
        if target_units is not None:     
            safe_target_units = max(1, target_units)
            progress_pct = 100.0 * units_seen / safe_target_units
            remaining_units = max(0, target_units - units_seen)
            eta_sec = remaining_units / units_per_sec if units_per_sec > 0 else float("inf")
            eta_text = f"{eta_sec:.1f}" if eta_sec != float("inf") else "unknown"
            msg = (
                f"{msg}, "               
                f"target_units={target_units} ({progress_pct:.2f}%), "
                f"eta_sec={eta_text}"
            )

        self._log(channel, msg)
        progress.last_log_at = now

    def _log_burn_completion(
        self,
        *,
        channel: str,
        label: str,
        progress: BurnContext,
        units_seen: int,
        examples_seen: int,
        target_units: int,
    ) -> None:
        elapsed_sec = max(0.0, time.perf_counter() - progress.started_at)
        units_per_sec = units_seen / elapsed_sec if elapsed_sec > 0 else 0.0
        self._log(
            channel,
            (
                f"{label} complete: "
                f"elapsed_sec={elapsed_sec:.1f}, "
                f"examples_seen={examples_seen}, "
                f"units_seen={units_seen}, "
                f"target_units={target_units}, "
                f"units_per_sec={units_per_sec:.1f}"
            ),
        )
