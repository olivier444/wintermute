# -*- coding: utf-8 -*-

from pathlib import Path

from wintermute.ml.inference.endpoint_config import InferenceEndpointConfig
from wintermute.ml.inference.slack_endpoint import SlackInferenceEndpoint
from wintermute.tools.slack.runtime import SlackInferenceRuntime


def start_inference_endpoint(output_root: str, endpoint_config_file: str) -> None:
    config = InferenceEndpointConfig.load(endpoint_config_file)
    slack = config.slack_config
    root = Path(output_root)

    runtime = SlackInferenceRuntime(
        bot_token=slack.bot_token,
        app_token=slack.app_token,
        channel=slack.channel,
    ).start()
    endpoint = SlackInferenceEndpoint(
        runtime=runtime,
        root=root,
        models=config.model_targets,
        default_model_name=config.default_model_name,
        device=config.device,
        system_prompt=config.system_prompt,
        poll_interval_sec=config.poll_interval_sec,
        session_log_dir=root / "inference_sessions",
    )

    try:
        endpoint.run_forever()
    finally:
        endpoint.close()
        runtime.close()
