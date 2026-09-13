from __future__ import annotations

# GENERIC
FLD_GENERIC_UID="r_uid"
FLD_GENERIC_SIGNATURE="signature"
FLD_GENERIC_TEXT = "text"
FLD_GENERIC_PROMPT = "prompt"
FLD_GENERIC_COMPLETION = "completion"
FLD_GENERIC_TASK = "task"
FLD_GENERIC_SCRATCHPAD = "scratchpad"
FLD_GENERIC_SYSTEM_PROMPT = "system_prompt"

# TASK LABELS
# Annotate the requested output behavior, not the subject domain or the amount of
# implicit difficulty. Use the primary supervised behavior when a prompt combines
# several operations; omit the label when no single behavior is reliably dominant.

# Translate source content from one natural language into another while preserving
# its meaning, tone, and relevant detail. Use when cross-language transfer is the
# primary requested operation, including bidirectional translation tasks.
TASK_TRANSLATION = "translation"

# Condense supplied source content into a shorter faithful account of its central
# information, such as a summary, synopsis, abstract, digest, or TL;DR. The output
# must primarily compress existing content; title/headline writing is generation.
TASK_SUMMARIZATION = "summarization"

# Transform supplied text while preserving its core meaning, without changing its
# language or primarily shortening it. This covers paraphrasing, correction, style
# transfer, simplification, controlled expansion, reformatting, and other editing.
TASK_REWRITING = "rewriting"

# Select one or more semantic classes from a predefined label set based on the
# input, including topic, intent, sentiment, entailment, or equivalence labels.
# Do not use for multiple-choice questions whose options are candidate answers.
TASK_CLASSIFICATION = "classification"

# Locate, assemble, reformulate, or structure facts explicitly present in supplied
# authoritative content. This includes extractive reading comprehension and fact
# extraction into schemas such as JSON; deriving a new conclusion is problem solving.
TASK_INFORMATION_EXTRACTION = "information_extraction"

# Answer a factual or explanatory request primarily from the model's parametric
# knowledge because no supplied source contains the authoritative answer. The prompt
# need not be phrased as a question; derivations and calculations are problem solving.
TASK_KNOWLEDGE_QA = "knowledge_qa"

# Create substantially new content from a specification, including drafting,
# brainstorming, planning, creative writing, code generation, and generating
# questions, contexts, passages, titles, or headlines. Do not use as a fallback
# for records whose task intent is unknown or genuinely mixed.
TASK_GENERATION = "generation"

# Construct a new solution or conclusion through calculation, deduction, diagnosis,
# proof, causal analysis, or another multi-step procedure. Use whether the reasoning
# trace is explicit or only the final solution is emitted; locating stated facts is
# information extraction, and recalling a fact is knowledge QA.
TASK_PROBLEM_SOLVING = "problem_solving"

TASK_LABELS = frozenset(
    {
        TASK_TRANSLATION,
        TASK_SUMMARIZATION,
        TASK_REWRITING,
        TASK_CLASSIFICATION,
        TASK_INFORMATION_EXTRACTION,
        TASK_KNOWLEDGE_QA,
        TASK_GENERATION,
        TASK_PROBLEM_SOLVING,
    }
)

# PATHS / DIRECTORIES / FILES
DIR_INDEX = "index"
DIR_RAW = "raw"
DIR_TOKENIZER = "tokenizer"
DIR_MATERIALIZED = "materialized"
DIR_EXTERNAL_MODELS = "external_models"
DIR_HUGGING_FACE_MODELS = "models"
DEFAULT_DATA_FILENAME = "data"

# INDEX TABLES
TBL_SOURCE = "source"
FLD_SOURCE_ID = "src_id"
FLD_SOURCE_LABEL = "src_label"
FLD_SOURCE_DATASET = "src_dataset"
FLD_SOURCE_CONFIG = "src_config"
FLD_SOURCE_REVISION = "src_revision"
FLD_SOURCE_SPLIT = "src_split"
FLD_SOURCE_TXT_FIELDS = "src_txt_fields"
FLD_SOURCE_LAN_FIELD = "src_lan_field"

TBL_FILE = "file"
FLD_FILE_ID = "fle_id"
FLD_FILE_SOURCE_ID = FLD_SOURCE_ID
FLD_FILE_NAME = "fle_name"
FLD_FILE_PATH = "fle_path"

TBL_RECORD = "record"
FLD_RECORD_ID = "rec_id"
FLD_RECORD_FILE_ID = FLD_FILE_ID
FLD_RECORD_LAN = "rec_lan"
FLD_RECORD_LEN = "rec_len"
FLD_RECORD_SNIPPET = "rec_sni"

TBL_SIGNATURE = "signature"
FLD_SIGNATURE_RECORD_ID=FLD_RECORD_ID
FLD_SIGNATURE_VECTOR="sig_vector"

TBL_BAND = "band"
FLD_BAND_RECORD_ID = FLD_RECORD_ID
FLD_BAND_ID = "bnd_id"
FLD_BAND_HASH = "bnd_hash"

TBL_SNAPSHOT = "snapshot"
FLD_SNAPSHOT_ID = "snp_id"

TBL_SNAPSHOT_MEMBERS = "snapshot_members"
FLD_MEMBERS_SNAPSHOT_ID = FLD_SNAPSHOT_ID
FLD_MEMBERS_REC_ID = FLD_RECORD_ID
FLD_MEMBERS_SPLIT = "split"
FLD_MEMBERS_HASH = "h"
FLD_MEMBERS_OVERSAMPLING = "oversampling"
