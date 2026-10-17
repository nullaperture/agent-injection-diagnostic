"""Explicit Qwen3.6 profile using the validated Qwen3_5 runtime interface."""
import importlib.metadata
import json
import os
from pathlib import Path

from .qwen import QwenRuntime, VERSIONS, require_remote, environment, digest

MODEL = 'Qwen/Qwen3.6-27B'
REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
PROFILE = Path(__file__).parent / 'profiles/qwen3.6-27b'

def manifest():
    value = json.loads((PROFILE / 'model-manifest.json').read_text())
    if (value['id'], value['revision']) != (MODEL, REVISION):
        raise ValueError('Wrong Qwen3.6 identity')
    return value

def describe():
    value = manifest()
    for name in value['bundled']:
        row = next(row for row in value['files'] if row['path'] == name)
        if digest(PROFILE / name) != row['sha256']:
            raise ValueError('Changed Qwen3.6 metadata: ' + name)
    config = json.loads((PROFILE / 'config.json').read_text())
    generation = json.loads((PROFILE / 'generation_config.json').read_text())
    if (config['architectures'] != ['Qwen3_5ForConditionalGeneration'] or
            (config['text_config']['hidden_size'], config['text_config']['num_hidden_layers']) != (5120, 64)
            or generation['eos_token_id'] != [248046, 248044] or generation['pad_token_id'] != 248044):
        raise ValueError('Qwen3.6 cannot use the shared native architecture/EOS contract')
    return {'model': MODEL, 'revision': REVISION, 'class': 'Qwen3_5ForConditionalGeneration',
        'backend': 'transformers', 'text_only': True, 'hidden_size': 5120, 'text_layers': 64,
        'runtime_versions': VERSIONS, 'dtype': 'bfloat16', 'attention': 'eager',
        'layer_path': 'model.language_model.layers', 'checkpoint_bytes': value['total_bytes'],
        'not_validated': ['defense transfer', 'tuning', 'engine parity', 'vision']}

def load_qwen36(seed=20261016):
    describe()
    require_remote()
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise RuntimeError('Qwen3.6 runtime version mismatch: ' + name)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one remote GPU required')
    if torch.cuda.get_device_properties(0).total_memory < 79 * 1024**3:
        raise RuntimeError('80GB-class GPU required')
    if os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') < 96 * 1000**3:
        raise RuntimeError('Host RAM below 96GB')
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'evaluation-sample/models'
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, token=False, cache_dir=cache,
                                     allow_patterns=[row['path'] for row in manifest()['files']]))
    acquired = {}
    for row in manifest()['files']:
        if Path(row['path']).name != row['path']:
            raise ValueError('Unsafe Qwen3.6 artifact path')
        path = snapshot / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Qwen3.6 acquisition mismatch: ' + row['path'])
        acquired[row['path']] = row
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, token=False, dtype=torch.bfloat16, attn_implementation='eager',
        device_map='cuda', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('Incomplete Qwen3.6 checkpoint load')
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    tokenizer.chat_template = (PROFILE / 'chat_template.jinja').read_text()
    if tokenizer.pad_token_id != 248044:
        raise RuntimeError('Unexpected Qwen3.6 tokenizer padding')
    runtime = QwenRuntime(torch, model, tokenizer)
    runtime.audit = {'profile': describe(), 'environment': environment(torch), 'acquisition': acquired,
        'loading': loading, 'seed': seed, 'deterministic_algorithms': True, 'allow_tf32': False,
        'base_parameters_frozen': True}
    runtime.layers()
    return runtime
