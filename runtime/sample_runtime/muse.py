"""Pinned remote text-only Muse adapter; preserves native ATEM/channel syntax."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .qwen import require_remote, environment, digest

PROFILE = Path(__file__).parent / 'profiles/muse-glimmer-30b'
MODEL = 'meta-models/Muse-Glimmer-30B'
REVISION = 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68'
VERSIONS = {'torch': '2.9.1', 'transformers': '5.20.1', 'accelerate': '1.12.0'}

def manifest():
    value = json.loads((PROFILE / 'model-manifest.json').read_text())
    if (value['id'], value['revision']) != (MODEL, REVISION):
        raise ValueError('Wrong Muse identity')
    return value

def describe():
    value = manifest()
    for name in value['bundled']:
        row = next(row for row in value['files'] if row['path'] == name)
        if digest(PROFILE / name) != row['sha256']:
            raise ValueError('Changed Muse bundled metadata: ' + name)
    return {'model': MODEL, 'revision': REVISION, 'backend': 'transformers', 'text_only': True,
            'class': 'MuseGlimmerForConditionalGeneration', 'hidden_size': 6656, 'text_layers': 52,
            'layer_path': 'model.language_model.layers', 'runtime_versions': VERSIONS,
            'dtype': 'bfloat16', 'attention': 'eager', 'checkpoint_bytes': value['total_bytes'],
            'not_validated': ['tuning', 'engine parity', 'vision', 'defense transfer']}

def verify_snapshot(directory):
    verified = {}
    for row in manifest()['files']:
        if Path(row['path']).name != row['path']:
            raise ValueError('Unsafe snapshot path')
        path = Path(directory) / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Muse acquisition mismatch: ' + row['path'])
        verified[row['path']] = dict(row)
    return verified

def load_muse(seed=20261016):
    describe()
    require_remote()
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise RuntimeError('Muse runtime version mismatch: ' + name)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one remote CUDA GPU required')
    if torch.cuda.get_device_properties(0).total_memory < 79 * 1024**3:
        raise RuntimeError('80GB-class GPU required')
    if os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') < 96 * 1000**3:
        raise RuntimeError('Host RAM below 96GB')
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'evaluation-sample/models'
    snapshot = snapshot_download(MODEL, revision=REVISION, token=False, cache_dir=cache,
                                 allow_patterns=[row['path'] for row in manifest()['files']])
    acquired = verify_snapshot(snapshot)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loading = MuseGlimmerForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, token=False,
        dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('Incomplete Muse checkpoint load')
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    tokenizer.chat_template = (PROFILE / 'chat_template.jinja').read_text()
    expected = {'<|begin_of_text|>': 200000, '<|end_of_text|>': 200001, '<|eom|>': 200007,
                '<|eot|>': 200008, '<|finetune_right_pad|>': 200018}
    if any(tokenizer.convert_tokens_to_ids(token) != identity for token, identity in expected.items()):
        raise RuntimeError('Unexpected Muse native control tokens')
    runtime = MuseRuntime(torch, model, tokenizer)
    runtime.audit = {'profile': describe(), 'environment': environment(torch), 'loading': loading,
                     'acquisition': acquired, 'seed': seed, 'deterministic_algorithms': True,
                     'allow_tf32': False, 'base_parameters_frozen': True}
    runtime.layers()
    return runtime

class MuseRuntime:
    backend_id = 'transformers'

    def __init__(self, torch, model, tokenizer):
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.audit = {}

    def layers(self):
        layers = self.model.model.language_model.layers
        if len(layers) != 52 or self.model.config.text_config.hidden_size != 6656:
            raise ValueError('Unexpected Muse text layers')
        return layers

    def render(self, messages, *, tools=None, thinking=True, reasoning_effort='low'):
        if thinking is not True or reasoning_effort != 'low':
            raise ValueError('Muse comparison explicitly uses native low reasoning; no invented thinking-off mode')
        if any(not isinstance(message.get('content', ''), str) for message in messages):
            raise ValueError('Text-only messages required')
        return self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
            add_generation_prompt=True, reasoning_strength='low', current_date='2026-10-16')

    def generate(self, messages, *, tools=None, thinking=True, reasoning_effort='low',
                 max_new_tokens=4096, max_input_tokens=8192):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
            raise ValueError('Invalid generation cap')
        if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 16384:
            raise ValueError('Invalid input cap')
        prompt = self.render(messages, tools=tools, thinking=thinking, reasoning_effort=reasoning_effort)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        width = encoded['input_ids'].shape[1]
        if width > max_input_tokens:
            raise ValueError('Input exceeds cap; nothing truncated')
        inputs = {key: value.to(self.model.device) for key, value in encoded.items()}
        from transformers import GenerationConfig
        config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens,
            use_cache=True, eos_token_id=[200001, 200008], pad_token_id=200018, bos_token_id=200000)
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, generation_config=config)
        self.torch.cuda.synchronize()
        ids = generated[0].tolist()
        if ids[:width] != inputs['input_ids'][0].tolist():
            raise RuntimeError('Muse generation prefix mismatch')
        tail = ids[width:]
        eos = bool(tail and tail[-1] in (200001, 200008))
        return {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt_token_ids': ids[:width],
            'generated_token_ids': tail, 'raw_completion': self.tokenizer.decode(tail, skip_special_tokens=False),
            'completion': self.tokenizer.decode(tail[:-1] if eos else tail, skip_special_tokens=False),
            'completion_decode_policy': 'remove observed final EOS only; retain native channels',
            'finish': 'eos' if eos else ('cap_without_eos' if len(tail) == max_new_tokens else 'unknown'),
            'generation_config': config.to_dict(), 'input_tokens': width, 'thinking': thinking,
            'reasoning_effort': reasoning_effort, 'max_input_tokens': max_input_tokens,
            'elapsed_seconds': time.perf_counter() - start, 'tools_executed': False}
