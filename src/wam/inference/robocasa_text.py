"""Training-compatible T5 conditioning for dynamic RoboCasa instructions."""
from collections import OrderedDict
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import torch

from wam.datasets.robocasa_posttrain import text_name
from wam.datasets.robocasa_visual_cache import require

LOG = logging.getLogger(__name__)


class RoboCasaInferenceText:
    def __init__(self, training_cache, model_config, device, *, cache_dir=None):
        # Optional old cache support is retained for callers outside the public
        # policy entry. Public inference always passes None and encodes online.
        self.cache = training_cache
        self.model_config = dict(model_config)
        self.device = torch.device(device)
        require((self.cache is None or self.cache.manifest["context_len"] == 128)
                and self.model_config.get("tokenizer_max_len", 128) == 128,
                "RoboCasa online T5 requires the training context length 128")
        self.encoder = self.tokenizer = None
        self.values = OrderedDict()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self._resource_configs = None
        self._identity = None

    def _resources(self):
        if self._resource_configs is None:
            from wam.model.backbones.wan22.helpers.loader import _resolve_configs
            cfg = self.model_config
            _, text, _, tokenizer = _resolve_configs(
                cfg['model_id'], cfg['tokenizer_model_id'], cfg.get('redirect_common_files', True))
            text.download_if_necessary()
            tokenizer.download_if_necessary()
            require(Path(text.path).is_file() and Path(tokenizer.path).is_dir(), 'Missing T5 resources')
            self._resource_configs = text, tokenizer
        return self._resource_configs

    def _cache_identity(self):
        if self._identity is None:
            text, tokenizer = self._resources()

            def identity(path):
                path = Path(path).resolve(strict=True)
                stat = path.stat()
                # Cache invalidation only; not a content-integrity/hash gate.
                return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

            self._identity = {
                'schema': 'phoenix.robocasa.online_t5.v1',
                'model_id': self.model_config['model_id'],
                'tokenizer_model_id': self.model_config['tokenizer_model_id'],
                'text': identity(text.path),
                'tokenizer': [identity(p) for p in sorted(Path(tokenizer.path).rglob('*')) if p.is_file()],
                'torch': str(torch.__version__), 'device_type': self.device.type,
                'device_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu',
                'cuda_version': torch.version.cuda,
                'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'dtype': 'bfloat16', 'context_len': 128, 'context_dim': 4096,
                'padding': 'zero_context_full_attention', 'tokenizer_clean': 'whitespace',
            }
        return self._identity

    def _load_encoder(self):
        from wam.model.backbones.wan22.helpers.loader import _load_registered_model
        from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        text, tokenizer = self._resources()
        tokenizer_path = Path(tokenizer.path)
        require(Path(text.path).is_file() and tokenizer_path.is_dir(), "Missing T5 resources")
        # Model construction consumes CPU RNG; preserve the policy's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            encoder = _load_registered_model(text.path, "wan_video_text_encoder",
                torch_dtype=torch.bfloat16, device=str(self.device)).eval().requires_grad_(False)
        tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=128, clean="whitespace")
        self.encoder, self.tokenizer = encoder, tokenizer
        LOG.info("Loaded online T5 on %s (file hash validation disabled)", self.device)

    @torch.inference_mode()
    def _encode(self, prompt):
        if self.encoder is None:
            self._load_encoder()
        ids, mask = self.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        context = self.encoder(ids.to(self.device), mask.to(device=self.device, dtype=torch.bool))[0].cpu()
        return context, mask[0].bool().cpu()

    @staticmethod
    def _validate(context, mask):
        require(isinstance(context, torch.Tensor) and isinstance(mask, torch.Tensor)
                and context.shape == (128, 4096) and context.dtype == torch.bfloat16
                and mask.shape == (128,) and mask.dtype == torch.bool
                and bool(mask.any()) and bool(torch.isfinite(context).all()), 'Invalid online T5 encoding')

    def _online(self, prompt):
        context, mask = self._encode(prompt)
        self._validate(context, mask)
        LOG.info('Online T5 encoded actual prompt: %s', prompt)
        return context.masked_fill(~mask[:, None], 0), torch.ones_like(mask)

    def _persistent(self, prompt):
        identity = self._cache_identity()
        key = hashlib.sha256(json.dumps({'identity': identity, 'prompt': prompt},
                                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (key + '.pt')
        # One writer per exact prompt/model/platform; readers never see a partial
        # tensor. Locks and atomic rename work across local evaluation workers.
        with (self.cache_dir / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                obj = torch.load(path, map_location='cpu', weights_only=True)
                require(obj['identity'] == identity and obj['prompt'] == prompt,
                        'Online T5 cache identity mismatch')
                context, mask = obj['context'], obj['mask']
                self._validate(context, mask)
                require(bool(mask.all()), 'Online T5 cache must use full attention')
                return context, mask
            context, mask = self._online(prompt)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=key + '.', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    torch.save({'identity': identity, 'prompt': prompt, 'context': context, 'mask': mask}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return context, mask

    def get(self, prompt):
        require(isinstance(prompt, str) and bool(prompt.strip()), "Missing T5 prompt")
        # A missing/corrupted committed cache entry is an error, not a cache miss.
        if self.cache is not None and text_name(prompt) in self.cache.manifest["files"]:
            return self.cache.get(prompt)
        if prompt not in self.values:
            self.values[prompt] = self._persistent(prompt) if self.cache_dir is not None else self._online(prompt)
            if len(self.values) > 128:
                self.values.popitem(last=False)
        self.values.move_to_end(prompt)
        return tuple(value.clone() for value in self.values[prompt])
