"""Official CoOp + DePT adapted to this repository's Dassl entry point.

The implementation follows Koorye/DePT's ``elp_coop.py``. DePT isolates
base-specific knowledge in a FiLM-conditioned linear space. New-class
inference deliberately uses only CoOp cosine-similarity logits.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler
from dassl.utils import load_checkpoint, load_pretrained_weights

from trainers.coop import CoOp, CustomCLIP as CoOpCustomCLIP, load_clip_to_cpu
from trainers.optim import build_optimizer


MODEL_NAME = "dept_model"


class FiLM(nn.Module):
    """Channel-wise transformation (cwT) used by the official DePT code."""

    def __init__(self, dim: int, bias: bool = True, use_sigmoid: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.use_sigmoid = use_sigmoid

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.scale.unsqueeze(0).to(dtype=value.dtype, device=value.device)
        output = scale * value
        if self.bias is not None:
            bias = self.bias.unsqueeze(0).to(dtype=value.dtype, device=value.device)
            output = output + bias
        return output.sigmoid() if self.use_sigmoid else output


class DePTCustomCLIP(CoOpCustomCLIP):
    """CoOp host with DePT's isolated base-specific feature space."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = str(cfg.DATASET.SUBSAMPLE_CLASSES)
        self.dataset_name = str(cfg.DATASET.NAME)
        self.lp_cfg = cfg.TRAINER.LINEAR_PROBE
        self.film_cfg = cfg.TRAINER.FILM

        feature_dim = int(clip_model.text_projection.size(1))
        if self.film_cfg.LINEAR_PROBE:
            self.film_lp_img = FiLM(feature_dim)
            self.film_lp_text = FiLM(feature_dim)

        if self.is_base_split:
            if self.lp_cfg.TYPE not in ("similarity", "linear"):
                raise ValueError(f"unsupported DePT linear-probe type: {self.lp_cfg.TYPE}")
            if self.lp_cfg.TYPE == "linear":
                self.linear_probe_proj = nn.Linear(feature_dim, len(classnames)).type(self.dtype)
            else:
                self.linear_probe_proj = nn.Identity()
        else:
            self.linear_probe_proj = nn.Identity()

    @property
    def is_base_split(self) -> bool:
        return self.subsample_classes == "base" or (
            self.subsample_classes == "all" and "ImageNet" in self.dataset_name
        )

    def encode_features(
        self,
        image: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        if image_features is None:
            image_features = self.image_encoder(image.type(self.dtype))
        return text_features, image_features

    def similarity_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        return self.logit_scale.exp() * image_features @ text_features.t()

    def linear_probe_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.film_cfg.LINEAR_PROBE:
            text_features = self.film_lp_text(text_features)
            image_features = self.film_lp_img(image_features)

        if self.lp_cfg.TYPE == "similarity":
            return self.similarity_logits(text_features, image_features), labels

        if labels is None:
            features = image_features
            output_labels = None
        else:
            features = torch.cat((text_features[labels], image_features), dim=0)
            output_labels = torch.cat((labels, labels), dim=0)
        return self.linear_probe_proj(features), output_labels

    def base_components(
        self,
        image: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ):
        text_features, image_features = self.encode_features(image, image_features)
        similarity = self.similarity_logits(text_features, image_features)
        linear, linear_labels = self.linear_probe_logits(
            text_features, image_features, labels
        )
        return similarity, linear, linear_labels, text_features

    def base_loss(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ):
        similarity, linear, linear_labels, text_features = self.base_components(
            image, labels, image_features
        )
        similarity_loss = F.cross_entropy(similarity, labels)
        linear_loss = F.cross_entropy(linear, linear_labels)
        weight = float(self.lp_cfg.WEIGHT)
        loss = (1.0 - weight) * similarity_loss + weight * linear_loss
        return loss, similarity_loss, linear_loss, similarity, text_features

    def forward(self, image: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if self.is_base_split:
            if labels is not None:
                return self.base_loss(image, labels)[0]
            similarity, linear, _, _ = self.base_components(image)
            if not self.lp_cfg.TEST_TIME_FUSION:
                return linear
            weight = float(self.lp_cfg.WEIGHT)
            return (1.0 - weight) * similarity + weight * linear

        if labels is not None:
            raise RuntimeError("DePT new-class inference must not receive training labels")
        text_features, image_features = self.encode_features(image)
        return self.similarity_logits(text_features, image_features)


def freeze_dept_parameters(model: nn.Module) -> Tuple[str, ...]:
    update_tokens = ("prompt_learner", "linear_probe", "film")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(token in name for token in update_tokens))
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if not trainable or not any("prompt_learner.ctx" in name for name in trainable):
        raise RuntimeError("DePT did not expose a trainable CoOp context")
    return trainable


def checkpoint_path(directory: str, model_name: str, epoch) -> str:
    model_directory = osp.join(directory, model_name)
    if epoch is None:
        filename = "model-best.pth.tar"
    elif int(epoch) >= 0:
        filename = f"model.pth.tar-{int(epoch)}"
    else:
        candidates = []
        for value in os.listdir(model_directory):
            prefix = "model.pth.tar-"
            if value.startswith(prefix) and value[len(prefix) :].isdigit():
                candidates.append((int(value[len(prefix) :]), value))
        if not candidates:
            raise FileNotFoundError(f'no DePT checkpoint found in "{model_directory}"')
        filename = max(candidates)[1]
    return osp.join(model_directory, filename)


@TRAINER_REGISTRY.register()
class DePT(CoOp):
    """Official CoOp+DePT with a frozen CLIP backbone."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ("fp16", "fp32", "amp")
        assert not cfg.TRAINER.COOP.CSC, "DePT uses the official generic context"
        assert int(cfg.OPTIM.MAX_EPOCH) == 10, "requested DePT protocol uses 10 epochs"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if torch.cuda.device_count() > 1:
            raise RuntimeError("DePT_CoOp supports one visible GPU only")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building official CoOp + DePT")
        self.model = DePTCustomCLIP(cfg, classnames, clip_model)
        trainable = freeze_dept_parameters(self.model)
        print(f"Parameters to be updated: {list(trainable)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)

        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)
        if infos is not None:
            for info in infos:
                print(f"DePT learning rate {info['lr']}: {info['layers']}")
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(MODEL_NAME, self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        if self.cfg.TRAINER.COOP.PREC == "amp":
            with autocast():
                loss = self.model(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = self.model(image, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return {"loss": float(loss.detach().item())}

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        path = checkpoint_path(directory, MODEL_NAME, epoch)
        checkpoint = load_checkpoint(path)
        state_dict = checkpoint["state_dict"]
        for key in (
            "prompt_learner.token_prefix",
            "prompt_learner.token_suffix",
        ):
            state_dict.pop(key, None)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "prompt_learner.token_prefix",
            "prompt_learner.token_suffix",
        }
        allowed_unexpected = {
            "linear_probe_proj.weight",
            "linear_probe_proj.bias",
        }
        missing = set(incompatible.missing_keys) - allowed_missing
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        if missing or unexpected:
            raise RuntimeError(
                "incompatible DePT checkpoint: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        print(
            f'Loaded DePT from "{path}" '
            f'(epoch = {checkpoint.get("epoch", "unknown")})'
        )


# Backward-compatible source name used by the reference implementation.
DePT_CoOp = DePT


__all__ = [
    "DePT",
    "DePT_CoOp",
    "DePTCustomCLIP",
    "FiLM",
    "MODEL_NAME",
    "checkpoint_path",
    "freeze_dept_parameters",
]
