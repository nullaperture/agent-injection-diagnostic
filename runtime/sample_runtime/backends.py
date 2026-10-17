"""Backend boundaries and offline handoff checks; no engine imports or execution.

A model identity is independent of its serving engine. This module does not
assert cross-engine equivalence, engine support, or a portable activation hook.
"""
from typing import Protocol


class InferenceBackend(Protocol):
    backend_id: str

    def generate(self, messages, *, tools=None, thinking=False, reasoning_effort="medium",
                 max_new_tokens=2048, max_input_tokens=8192) -> dict: ...


def validate_handoff(source, target):
    """Fail closed on silent model/adapter/template changes between pipeline stages.

    Records are data, not filenames to import or shell commands to execute.
    Artifact hashes refer to a separate acquired artifact manifest; this check
    compares declarations, not the artifact bytes or engine behavior.
    """
    for key in ("model", "revision", "tokenizer_sha256", "chat_template_sha256"):
        if not source.get(key) or source.get(key) != target.get(key):
            raise ValueError("Model or prompt identity changed across stages: " + key)
    for key in ("backend", "backend_version", "dtype", "quantization", "thinking", "reasoning_effort"):
        if key not in source or key not in target:
            raise ValueError("Explicit backend/runtime settings required: " + key)
    kind = source.get("artifact_kind", "base")
    if kind not in ("base", "activation-hook", "lora", "checkpoint"):
        raise ValueError("Unknown handoff artifact kind")
    if kind != target.get("artifact_kind", "base"):
        raise ValueError("A transformed model cannot silently become the base model")
    if kind != "base":
        if not source.get("artifact_manifest_sha256") or source["artifact_manifest_sha256"] != target.get("artifact_manifest_sha256"):
            raise ValueError("Missing or changed transformation artifact manifest")
    if kind == "activation-hook" and source["backend"] != target["backend"]:
        raise ValueError("In-memory activation hook is not portable to another engine")
    if kind in ("lora", "checkpoint") and target.get("artifact_load_verified") is not True:
        raise ValueError("Target must verify loading the exact changed artifact")
    changed = [key for key in ("backend", "backend_version", "dtype", "quantization", "thinking", "reasoning_effort")
               if source[key] != target[key]]
    return {"declaration_compatible": True, "changed_settings": changed,
            "parity_status": "requires measured comparison" if changed else "not established by declarations",
            "artifact_bytes_verified_by_this_check": False, "engine_executed": False}
