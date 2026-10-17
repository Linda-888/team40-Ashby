"""One interface over CONCH, MUSK, PathGen-CLIP, QuiltNet and PLIP.

Every encoder returns L2-normalized embeddings in the model's shared image-text space
(the projected / contrastive head), because Test 2 compares images to text directly.
Imports are lazy: MUSK lives in its own env (see setup_env.sh).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Encoder:
    name = ""
    max_words = 60  # rough text chunk size that fits the model's token limit

    def __init__(self, device: str):
        self.device = device

    def preprocess(self, tile: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> float (3, h, w)."""
        return self.transform(Image.fromarray(tile))

    def _images(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def embed_images(self, x: torch.Tensor) -> np.ndarray:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            z = self._images(x.to(self.device))
        return F.normalize(z.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                z = self._texts(texts[i:i + batch_size])
            out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def embed_tiles(self, tiles: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Convenience for small jobs: uint8 (N, H, W, 3) -> (N, D)."""
        out = []
        for i in range(0, len(tiles), batch_size):
            x = torch.stack([self.preprocess(t) for t in tiles[i:i + batch_size]])
            out.append(self.embed_images(x))
        return np.concatenate(out)


class Conch(Encoder):
    name = "conch"
    max_words = 90  # 127 BPE tokens

    def __init__(self, device):
        super().__init__(device)
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

        # The hackathon pre-loads weights on Delta: point TF_CONCH_CKPT at that pytorch_model.bin.
        ckpt = _local_ckpt("TF_CONCH_CKPT", "pytorch_model.bin") or "hf_hub:MahmoodLab/conch"
        self.model, self.transform = create_model_from_pretrained(
            "conch_ViT-B-16", ckpt, hf_auth_token=os.environ.get("HF_TOKEN"))
        self.model.to(device).eval()
        self.tokenizer = get_tokenizer()
        self._tokenize = tokenize

    def _images(self, x):
        return self.model.encode_image(x, proj_contrast=True, normalize=True)

    def _texts(self, texts):
        tokens = self._tokenize(tokenizer=self.tokenizer, texts=texts).to(self.device)
        return self.model.encode_text(tokens, normalize=True)


class Musk(Encoder):
    name = "musk"
    max_words = 70  # 100 XLM-R tokens

    def __init__(self, device):
        super().__init__(device)
        import torchvision.transforms as T
        from musk import modeling, utils  # noqa: F401  (modeling registers the timm model)
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from timm.models import create_model
        from transformers import XLMRobertaTokenizer

        self.model = create_model("musk_large_patch16_384")
        # Pre-loaded weights (TF_MUSK_CKPT, a .safetensors file) win; otherwise download from the hub
        # into a shared folder (setup_env.sh puts model.safetensors there).
        ckpt = _local_ckpt("TF_MUSK_CKPT", "model.safetensors") or "hf_hub:xiangjx/musk"
        local_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "musk") + "/"
        utils.load_model_and_may_interpolate(ckpt, self.model, "model|module", "", local_dir=local_dir)
        self.model.to(device=device, dtype=torch.float16).eval()
        self.transform = T.Compose([
            T.Resize(384, interpolation=3, antialias=True),
            T.CenterCrop((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])
        self.tokenizer = XLMRobertaTokenizer(os.path.join(os.path.dirname(utils.__file__), "models", "tokenizer.spm"))
        self._xlm_tokenizer = utils.xlm_tokenizer

    def _images(self, x):
        # with_head + out_norm puts images in the image-text space; ms_aug is for linear probes only.
        return self.model(image=x.to(dtype=torch.float16), with_head=True, out_norm=True,
                          ms_aug=False, return_global=True)[0]

    def _texts(self, texts):
        ids, pads = [], []
        for t in texts:
            i, p = self._xlm_tokenizer(t, self.tokenizer, max_len=100)
            ids.append(torch.tensor(i).unsqueeze(0))
            pads.append(torch.tensor(p).unsqueeze(0))
        return self.model(text_description=torch.cat(ids).to(self.device),
                          padding_mask=torch.cat(pads).to(self.device),
                          with_head=True, out_norm=True, ms_aug=False, return_global=True)[1]


class _OpenClip(Encoder):
    max_words = 45  # CLIP's 77 tokens

    def __init__(self, device, model_name, pretrained=None):
        super().__init__(device)
        import open_clip

        self.model, _, self.transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _images(self, x):
        return self.model.encode_image(x)

    def _texts(self, texts):
        return self.model.encode_text(self.tokenizer(texts).to(self.device))


class PathGen(_OpenClip):
    name = "pathgen"

    def __init__(self, device):
        super().__init__(device, "ViT-B-16", os.environ.get("TF_PATHGEN_CKPT", "pathgenclip.pt"))


class Quilt(_OpenClip):
    """QuiltNet-B-32: ungated, a backup while CONCH/MUSK access is pending."""
    name = "quilt"

    def __init__(self, device):
        super().__init__(device, "hf-hub:wisdomik/QuiltNet-B-32")


class _HFImageTransform:
    """PIL -> (3, h, w) tensor via a transformers image processor (picklable for DataLoader workers)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


class Plip(Encoder):
    """PLIP (CLIP fine-tuned on OpenPath): ungated."""
    name = "plip"
    max_words = 45

    def __init__(self, device):
        super().__init__(device)
        from transformers import CLIPModel, CLIPProcessor

        repo = os.environ.get("TF_PLIP_CKPT", "vinid/plip")
        self.model = CLIPModel.from_pretrained(repo).to(device).eval()
        processor = CLIPProcessor.from_pretrained(repo)
        self.transform = _HFImageTransform(processor.image_processor)
        self.tokenizer = processor.tokenizer

    def _images(self, x):
        return self.model.get_image_features(pixel_values=x)

    def _texts(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        return self.model.get_text_features(**tok.to(self.device))


def _local_ckpt(env_var: str, filename: str):
    """Path from env_var if set; a directory means <dir>/<filename>."""
    path = os.environ.get(env_var)
    if not path:
        return None
    path = os.path.join(path, filename) if os.path.isdir(path) else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{env_var}={path} does not exist")
    return path


ENCODERS = {"conch": Conch, "musk": Musk, "pathgen": PathGen, "quilt": Quilt, "plip": Plip}


def load_encoder(name: str, device: str | None = None) -> Encoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return ENCODERS[name](device)
