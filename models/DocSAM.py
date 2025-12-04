# models/DocSAM.py
"""
DocSAM wrapper (inferência-friendly)

Características:
- Usa implementação LOCAL de Mask2Former (se existir em models/mask2former) ou HuggingFace como fallback.
- forward() para inferência: aceita apenas {"pixel_values": Tensor, "pixel_mask": Tensor (opcional)}.
- load_docsam_checkpoint(path, strict=False): carrega checkpoints .pth com tolerância a prefixos.
- predict_image(image, resize=(1024,1024)): conveniência para rodar inferência em PIL.Image.
"""

import os
import inspect
import types
import torch
import torch.nn as nn
from torch.nn import functional as F

# optional: use HF processor if available for consistent preprocessing/postprocessing
try:
    from transformers import AutoImageProcessor, Mask2FormerConfig
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Try a local Mask2Former implementation in repository (common in forks)
def _try_import_local_mask2former():
    candidates = [
        "models.mask2former.modeling_mask2former",
        "models.mask2former.model",
        "models.mask2former.mask2former",
    ]
    for module_path in candidates:
        try:
            mod = __import__(module_path, fromlist=["Mask2FormerForUniversalSegmentation"])
            cls = getattr(mod, "Mask2FormerForUniversalSegmentation", None)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


class DocSAM(nn.Module):
    def __init__(self, model_size="base", mask2former_path: str = None, device=None):
        """
        Args:
            model_size: "base" or "large" (only for default path selection)
            mask2former_path: local folder path or HuggingFace repo id. If None, uses default local paths.
            device: "cpu" or "cuda" etc. If None, model returned on CPU.
        """
        super().__init__()

        # default mask2former folders (local)
        if mask2former_path is None:
            if model_size == "base":
                mask2former_path = "./pretrained_model/mask2former-swin-base-coco-panoptic"
            elif model_size == "large":
                mask2former_path = "./pretrained_model/mask2former-swin-large-coco-panoptic"
            else:
                mask2former_path = "./pretrained_model/mask2former-swin-base-coco-panoptic"

        self.mask2former_path = mask2former_path
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Try local impl
        LocalMask2Former = _try_import_local_mask2former()
        if LocalMask2Former is not None:
            Mask2FormerClass = LocalMask2Former
            self._using_local_impl = True
            print("DocSAM: using local Mask2Former implementation from repo.")
        else:
            if not HF_AVAILABLE:
                raise RuntimeError("Neither local Mask2Former found nor transformers available.")
            from transformers import Mask2FormerForUniversalSegmentation as HFMask2Former
            Mask2FormerClass = HFMask2Former
            self._using_local_impl = False
            print("DocSAM: using HuggingFace Mask2Former implementation as fallback.")

        # Load config if HF available and path is local folder
        self.config = None
        if HF_AVAILABLE:
            try:
                # If path is a local folder with config.json, load config from there
                if os.path.isdir(mask2former_path):
                    try:
                        self.config = Mask2FormerConfig.from_pretrained(mask2former_path, local_files_only=True)
                    except Exception:
                        # fallback: instantiate default config if loading fails
                        self.config = None
                else:
                    try:
                        self.config = Mask2FormerConfig.from_pretrained(mask2former_path)
                    except Exception:
                        self.config = None
            except Exception:
                self.config = None

        # Instantiate model (prefer using from_pretrained if folder exists)
        try:
            if os.path.isdir(mask2former_path):
                try:
                    # some local classes implement from_pretrained
                    self.mask2former = Mask2FormerClass.from_pretrained(mask2former_path, config=self.config) \
                        if hasattr(Mask2FormerClass, "from_pretrained") else Mask2FormerClass(self.config)
                except Exception:
                    # fallback instantiate
                    self.mask2former = Mask2FormerClass(self.config) if self.config is not None else Mask2FormerClass()
            else:
                # treat as HF repo id or instantiate from config
                if self.config is not None:
                    self.mask2former = Mask2FormerClass.from_pretrained(mask2former_path, config=self.config)
                else:
                    self.mask2former = Mask2FormerClass(self.config) if self.config is not None else Mask2FormerClass()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Mask2Former class: {e}") from e

        # move to device
        try:
            self.to(self.device)
        except Exception:
            pass

        # optional HF processor if available
        if HF_AVAILABLE:
            try:
                self.processor = AutoImageProcessor.from_pretrained(mask2former_path)
            except Exception:
                self.processor = None
        else:
            self.processor = None

    # --- checkpoint loader helper ---
    def load_docsam_checkpoint(self, ckpt_path: str, strict: bool = False, map_location="cpu"):
        """
        Loads a checkpoint that was trained for DocSAM (docsam_*.pth).
        The loader tries a few strategies to map keys:
          - direct load_state_dict
          - if keys have prefixes (e.g. "mask2former."), strip prefix and reload
        Returns: (missing_keys, unexpected_keys)
        """
        state = torch.load(ckpt_path, map_location=map_location)
        # if checkpoint is a dict with 'state_dict' key (common), extract it
        if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
            if "state_dict" in state:
                sd = state["state_dict"]
            else:
                sd = state["model_state_dict"]
        else:
            sd = state

        # Try direct load
        try:
            msg = self.mask2former.load_state_dict(sd, strict=strict)
            print("Loaded checkpoint directly into mask2former; result:", msg)
            return msg
        except Exception:
            # Try to adapt keys: find keys that contain "mask2former" prefix or "model."
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                # common prefixes to strip
                for prefix in ["mask2former.", "model.", "module."]:
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        break
                # also strip "model.mask2former." etc.
                for prefix in ["model.mask2former.", "model.backbone.", "backbone."]:
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        break
                new_sd[new_k] = v

            try:
                msg = self.mask2former.load_state_dict(new_sd, strict=strict)
                print("Loaded checkpoint with prefix-stripped keys; result:", msg)
                return msg
            except Exception as e:
                # Last attempt: try to map only matching keys by size
                cur = dict(self.mask2former.named_parameters())
                cur.update({k: v for k, v in self.mask2former.named_buffers()})
                matched = {}
                for k, v in new_sd.items():
                    if k in cur and cur[k].size() == v.size():
                        matched[k] = v
                if len(matched) == 0:
                    raise RuntimeError("Could not match any keys from checkpoint to model parameters.") from e
                load_msg = self.mask2former.load_state_dict(matched, strict=False)
                print("Loaded subset of checkpoint by size match; result:", load_msg)
                return load_msg

    # --- forward for inference ---
    def forward(self, batch):
        """
        Forward method for inference.
        Accepts batch: dict with keys:
            - pixel_values (Tensor) [B,C,H,W]
            - pixel_mask   (Tensor) optional
        Returns a SimpleNamespace with:
            - pred_masks
            - pred_boxes
            - class_scores
            - raw_output (original output)
        """
        if not isinstance(batch, dict):
            raise TypeError("forward expects a dict batch")

        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("batch must contain 'pixel_values' tensor")

        pixel_mask = batch.get("pixel_mask", None)
        if pixel_mask is None:
            # create full mask (True=valid)
            pixel_mask = torch.ones((pixel_values.shape[0], 1, pixel_values.shape[2], pixel_values.shape[3]),
                                    dtype=torch.bool, device=pixel_values.device)

        # figure out accepted args
        try:
            sig = inspect.signature(self.mask2former.forward)
            accepted = set(sig.parameters.keys())
        except Exception:
            accepted = {"pixel_values", "pixel_mask"}

        call_kwargs = {}
        if "pixel_values" in accepted:
            call_kwargs["pixel_values"] = pixel_values.to(self.device)
        if "pixel_mask" in accepted:
            call_kwargs["pixel_mask"] = pixel_mask.to(self.device)

        # call
        outputs = self.mask2former(**call_kwargs)

        # helper to try many possible field names
        def _safe_get(obj, *names):
            if obj is None:
                return None
            for n in names:
                # dict-like
                try:
                    if isinstance(obj, dict) and n in obj:
                        return obj[n]
                except Exception:
                    pass
                # attribute-like
                try:
                    if hasattr(obj, n):
                        return getattr(obj, n)
                except Exception:
                    pass
            return None

        pred_masks = _safe_get(outputs,
                               "pred_masks",
                               "masks",
                               "pred_mask_logits",
                               "masks_logits",
                               "mask_queries_logits",
                               "logits")
        pred_boxes = _safe_get(outputs,
                               "pred_boxes",
                               "boxes",
                               "pred_bboxes",
                               "pred_boxes_xyxy",
                               "pred_boxes_xywh")
        class_scores = _safe_get(outputs,
                                 "class_scores",
                                 "pred_scores",
                                 "scores",
                                 "pred_logits",
                                 "logits")

        # return normalized namespace
        return types.SimpleNamespace(
            pred_masks=pred_masks,
            pred_boxes=pred_boxes,
            class_scores=class_scores,
            raw_output=outputs
        )

    def predict_image(self, image, resize=(1024, 1024), device=None):
        """
        Convenience method: accepts PIL.Image or tensor and runs forward returning normalized outputs.
        - image: PIL.Image or torch.Tensor (C,H,W) or (B,C,H,W)
        - resize: (H,W) to resize PIL image
        - device: optional device string
        """
        self.eval()
        if device is None:
            device = self.device
        else:
            device = torch.device(device)

        # Prepare tensor
        if isinstance(image, torch.Tensor):
            inp = image
            if inp.ndim == 3:
                inp = inp.unsqueeze(0)
        else:
            # PIL.Image
            try:
                from torchvision import transforms as T
                transform = T.Compose([T.Resize(resize), T.ToTensor()])
            except Exception:
                raise RuntimeError("Please install torchvision to use predict_image with PIL images.")
            inp = transform(image).unsqueeze(0)

        pixel_mask = torch.ones_like(inp[:, :1, :, :], dtype=torch.bool, device=device)
        inp = inp.to(device)

        with torch.no_grad():
            out = self({"pixel_values": inp, "pixel_mask": pixel_mask})
        return out
