"""Shared text-only Qwen adapter. No provider operations or experiment scoring."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROFILE = Path(__file__).parent / "profiles" / "qwen3.8-27b"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VERSIONS = {"torch": "2.9.1", "transformers": "5.9.0", "accelerate": "1.12.0"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest():
    value = json.loads((PROFILE / "model-manifest.json").read_text())
    if (value["id"], value["revision"]) != (MODEL, REVISION):
        raise ValueError("Unexpected profile identity")
    return value


def describe():
    value = manifest()
    for name in ("config.json", "generation_config.json", "chat_template.jinja"):
        expected = next(row for row in value["files"] if row["path"] == name)
        if digest(PROFILE / name) != expected["sha256"]:
            raise ValueError("Changed bundled metadata: " + name)
    return {"model": MODEL, "revision": REVISION, "text_only": True,
            "class": "Qwen3_5ForConditionalGeneration", "hidden_size": 5120,
            "text_layers": 64, "layer_path": "model.language_model.layers",
            "runtime_versions": VERSIONS, "dtype": "bfloat16", "attention": "eager",
            "checkpoint_bytes": value["total_bytes"], "minimum_vram_gib": 79,
            "minimum_host_ram_gb": 96, "target_execution": "remote Linux CUDA only",
            "implemented": ["pinned acquisition", "text inference", "native tool rendering",
                            "text-layer access", "audit capture"],
            "backend": "transformers", "engine_adapters": {"transformers": "implemented; inspect per-run validation evidence", "vllm": "planned, not implemented"},
            "not_validated": ["behavioral defense transfer",
                              "LoRA trainer", "full fine-tuning", "vision inputs"]}


def verify_snapshot(directory):
    """Content verification, including all eighteen shards. No target imports."""
    directory = Path(directory)
    verified = {}
    for row in manifest()["files"]:
        name = row["path"]
        if Path(name).name != name:
            raise ValueError("Unexpected manifest path")
        path = directory / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Acquisition mismatch: " + name)
        verified[name] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    return verified


def require_remote():
    # Explicit controller attestation is an operational guard, not authentication.
    if (sys.platform != "linux" or "microsoft" in platform.release().lower()
            or os.environ.get("SAMPLE_REMOTE_TRIAL") != "1"):
        raise RuntimeError("Target loading requires the approved remote Linux worker; no local/WSL inference")
    if any(any(part in name.upper() for part in ("TOKEN", "SECRET", "API_KEY", "PASSWORD"))
           for name in os.environ):
        raise RuntimeError("Worker environment contains credential-like variables")


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(torch):
    """Measured values only; absent tools remain explicit capture gaps."""
    return {"observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
            "kernel": platform.release(), "machine": platform.machine(),
            "cpu": platform.processor() or command_output(["lscpu"]),
            "host_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "packages": command_output([sys.executable, "-m", "pip", "freeze", "--all"]),
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                                   "--format=csv,noheader"]),
            "provider_image_region_price": "controller must supply; not inferred from GPU"}


def load_qwen(*, snapshot=None, seed=20261016):
    """One shared frozen base per process; native model is exposed for later fit recipes.

    A fitting recipe must explicitly unfreeze its own adapters/parameters and record
    its data, optimizer and checkpoint contract. No trainer/defense is implied here.
    """
    describe()
    require_remote()  # Reject workstation execution before importing torch or downloading.
    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise RuntimeError("Runtime version mismatch: " + name)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one remote CUDA GPU required; no CPU fallback")
    if torch.cuda.get_device_properties(0).total_memory < 79 * 1024**3:
        raise RuntimeError("80GB-class GPU required")
    if os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") < 96 * 1000**3:
        raise RuntimeError("Host RAM below 96 GB requirement")
    if snapshot is None:
        from huggingface_hub import snapshot_download
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "evaluation-sample" / "models"
        snapshot = snapshot_download(MODEL, revision=REVISION, token=False, cache_dir=cache,
                                     allow_patterns=[row["path"] for row in manifest()["files"]])
    acquired = verify_snapshot(snapshot)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, token=False,
        dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda", output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Incomplete or unexpected checkpoint load")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    tok.chat_template = (PROFILE / "chat_template.jinja").read_text()
    if tok.pad_token_id != 248044:
        raise RuntimeError("Unexpected tokenizer padding")
    runtime = QwenRuntime(torch, model, tok)
    runtime.audit = {"profile": describe(), "acquisition": acquired, "loading": loading,
                     "environment": environment(torch), "seed": seed,
                     "deterministic_algorithms": True, "allow_tf32": False,
                     "base_parameters_frozen": True}
    runtime.layers()  # Validate the TEXT decoder, never match the vision tower.
    return runtime


class QwenRuntime:
    backend_id = "transformers"

    def __init__(self, torch, model, tokenizer):
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.audit = {}

    def layers(self):
        config = self.model.config.text_config
        layers = self.model.model.language_model.layers
        if config.hidden_size != 5120 or config.num_hidden_layers != 64 or len(layers) != 64:
            raise ValueError("Wrong text architecture; do not reuse another model's adapters")
        return layers

    def render(self, messages, *, tools=None, thinking=False, reasoning_effort="medium"):
        if type(thinking) is not bool or reasoning_effort not in ("low", "medium", "xhigh"):
            raise ValueError("Explicit supported thinking settings required")
        for message in messages:
            if not isinstance(message.get("content", ""), str):
                raise ValueError("This adapter supports text-only messages")
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking, reasoning_effort=reasoning_effort, preserve_thinking=True)

    def generate(self, messages, *, tools=None, thinking=False, reasoning_effort="medium",
                 max_new_tokens=2048, max_input_tokens=8192):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
            raise ValueError("max_new_tokens must be 1..8192")
        if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 16384:
            raise ValueError("max_input_tokens must be 1..16384; no silent truncation")
        prompt = self.render(messages, tools=tools, thinking=thinking, reasoning_effort=reasoning_effort)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        width = encoded["input_ids"].shape[1]
        if width > max_input_tokens:
            raise ValueError("Prompt exceeds recorded input cap; nothing truncated or generated")
        inputs = {key: value.to(self.model.device) for key, value in encoded.items()}
        from transformers import GenerationConfig
        # Independent config avoids inheriting sampling or hidden model defaults.
        config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                                  max_new_tokens=max_new_tokens, use_cache=True,
                                  eos_token_id=[248046, 248044], pad_token_id=248044,
                                  bos_token_id=248044)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=config)
        self.torch.cuda.synchronize()
        ids = generated[0].tolist()
        if ids[:width] != inputs["input_ids"][0].tolist():
            raise ValueError("Generation did not preserve the input prefix")
        tail = ids[width:]
        stopped = bool(tail and tail[-1] in (248046, 248044))
        return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_ids": ids[:width], "generated_token_ids": tail,
                "raw_completion": self.tokenizer.decode(tail, skip_special_tokens=False),
                # Preserve reasoning/tool delimiters even if tokenizers mark them
                # special. Remove only the observed final EOS token by ID.
                "completion": self.tokenizer.decode(tail[:-1] if stopped else tail, skip_special_tokens=False),
                "completion_decode_policy": "remove observed final EOS only; retain other delimiters",
                "finish": "eos" if stopped else ("cap_without_eos" if len(tail) == max_new_tokens else "unknown"),
                "generation_config": config.to_dict(), "input_tokens": width,
                "thinking": thinking, "reasoning_effort": reasoning_effort,
                "max_input_tokens": max_input_tokens, "elapsed_seconds": time.perf_counter() - started,
                "tools_executed": False}
