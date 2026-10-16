"""Training-compatible T5 conditioning for dynamic RoboCasa instructions."""
from collections import OrderedDict
import logging
from pathlib import Path

import torch

from wam.datasets.robocasa_posttrain import text_name
from wam.datasets.robocasa_visual_cache import digest, file_hash, require

LOG = logging.getLogger(__name__)


class RoboCasaInferenceText:
    def __init__(self, training_cache, model_config, device):
        self.cache = training_cache
        self.model_config = dict(model_config)
        self.device = torch.device(device)
        require(self.cache.manifest["context_len"] == 128
                and self.model_config.get("tokenizer_max_len", 128) == 128,
                "RoboCasa online T5 requires the training context length 128")
        self.encoder = self.tokenizer = None
        self.values = OrderedDict()

    def _load_encoder(self):
        from wam.model.backbones.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        cfg = self.model_config
        _, text, _, tokenizer = _resolve_configs(
            cfg["model_id"], cfg["tokenizer_model_id"], cfg.get("redirect_common_files", True))
        text.download_if_necessary()
        tokenizer.download_if_necessary()
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

    def get(self, prompt):
        require(isinstance(prompt, str) and bool(prompt.strip()), "Missing T5 prompt")
        # A missing/corrupted committed cache entry is an error, not a cache miss.
        if text_name(prompt) in self.cache.manifest["files"]:
            return self.cache.get(prompt)
        if prompt not in self.values:
            context, mask = self._encode(prompt)
            require(context.shape == (128, 4096) and context.dtype == torch.bfloat16
                    and mask.shape == (128,) and mask.dtype == torch.bool
                    and bool(mask.any()) and bool(torch.isfinite(context).all()), "Invalid online T5 encoding")
            # Match the native cached convention: zero padded tokens, full attention.
            self.values[prompt] = context.masked_fill(~mask[:, None], 0), torch.ones_like(mask)
            if len(self.values) > 128:
                self.values.popitem(last=False)
            LOG.info("Online T5 encoded actual prompt: %s", prompt)
        self.values.move_to_end(prompt)
        return tuple(value.clone() for value in self.values[prompt])
