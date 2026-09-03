"""Shared frozen-template envelope geometry for FTEP host integrations."""

from __future__ import annotations

import gc
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from trainers.ftep_mmrl import TEMPLATE_ENVELOPES


def sphere_log(base, points):
    """Map normalized points to the tangent space at normalized base points."""
    cosine = (base * points).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = points - cosine * base
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    angle = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    scale = torch.where(
        tangent_norm > 1e-7,
        angle / tangent_norm.clamp_min(1e-12),
        torch.ones_like(tangent_norm),
    )
    return tangent * scale


class FrozenTemplateEnvelope(nn.Module):
    """Exact projection to a small class-wise convex hull in tangent space."""

    def __init__(self, base_points, vertices):
        super().__init__()
        self.register_buffer("base_points", base_points.float())
        self.register_buffer("vertices", vertices.float())
        self.faces = tuple(
            indices
            for size in range(1, vertices.shape[1] + 1)
            for indices in itertools.combinations(range(vertices.shape[1]), size)
        )

    @torch.no_grad()
    def project(self, tangent):
        class_count = self.vertices.shape[0]
        best_distance = torch.full(
            (class_count,), float("inf"), device=tangent.device
        )
        best_target = torch.zeros_like(tangent)
        best_support = torch.zeros(
            class_count, device=tangent.device, dtype=torch.long
        )
        for indices in self.faces:
            face = self.vertices[:, indices, :]
            face_size = len(indices)
            if face_size == 1:
                weights = torch.ones(
                    class_count, 1, device=tangent.device, dtype=tangent.dtype
                )
                valid = torch.ones(
                    class_count, device=tangent.device, dtype=torch.bool
                )
            else:
                gram = torch.einsum("cmd,cnd->cmn", face, face)
                linear = torch.einsum("cmd,cd->cm", face, tangent)
                kkt = torch.zeros(
                    class_count,
                    face_size + 1,
                    face_size + 1,
                    device=tangent.device,
                    dtype=tangent.dtype,
                )
                identity = torch.eye(
                    face_size, device=tangent.device, dtype=tangent.dtype
                ).unsqueeze(0)
                kkt[:, :face_size, :face_size] = gram + 1e-7 * identity
                kkt[:, :face_size, face_size] = 1.0
                kkt[:, face_size, :face_size] = 1.0
                rhs = torch.zeros(
                    class_count,
                    face_size + 1,
                    1,
                    device=tangent.device,
                    dtype=tangent.dtype,
                )
                rhs[:, :face_size, 0] = linear
                rhs[:, face_size, 0] = 1.0
                weights = torch.linalg.solve(kkt, rhs).squeeze(-1)[:, :face_size]
                valid = weights.min(dim=1).values >= -1e-5
                weights = weights.clamp_min(0.0)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

            target = torch.einsum("cm,cmd->cd", weights, face)
            distance = (tangent - target).square().sum(dim=1)
            update = valid & (distance < best_distance)
            best_distance = torch.where(update, distance, best_distance)
            best_target = torch.where(update.unsqueeze(1), target, best_target)
            support = (weights > 1e-5).sum(dim=1)
            best_support = torch.where(update, support, best_support)

        if not torch.isfinite(best_distance).all():
            raise RuntimeError("FTEP plugin projection found no valid convex face")
        return best_target, best_support

    def forward(self, text_features, return_support=False):
        with torch.cuda.amp.autocast(enabled=False):
            text = F.normalize(text_features.float(), dim=-1)
            tangent = sphere_log(self.base_points.float(), text)
            target, support = self.project(tangent.detach())
            loss = 0.5 * (tangent - target).square().sum(dim=1).mean()
        if return_support:
            return loss, support
        return loss


def compile_frozen_template_envelope(
    cfg,
    classnames,
    clip_module,
    clip_loader,
    canonical_template=None,
):
    """Compile FTEP on CPU so host and frozen teacher models never coexist on GPU."""
    dataset = cfg.DATASET.NAME
    templates = list(TEMPLATE_ENVELOPES[dataset])
    if canonical_template is not None:
        templates[0] = canonical_template

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    frozen_clip = None
    try:
        frozen_clip = clip_loader(cfg).float().eval()
        features = []
        with torch.no_grad():
            for template in templates:
                tokenized = torch.cat(
                    [
                        clip_module.tokenize(template.format(name.replace("_", " ")))
                        for name in classnames
                    ]
                )
                encoded = frozen_clip.encode_text(tokenized)
                features.append(F.normalize(encoded.float(), dim=-1))
    finally:
        del frozen_clip
        gc.collect()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    bank = torch.stack(features, dim=1)
    base_points = F.normalize(bank.mean(dim=1), dim=-1)
    vertices = sphere_log(base_points.unsqueeze(1), bank)
    radii = vertices.norm(dim=-1)
    projector = FrozenTemplateEnvelope(base_points, vertices)
    stats = {
        "template_count": len(templates),
        "mean_radius_degrees": float(radii.mean().item() * 180.0 / math.pi),
        "max_radius_degrees": float(radii.max().item() * 180.0 / math.pi),
    }
    return projector, stats


def host_text_features(model):
    """Return normalized learned text features through the host's own text path."""
    host = model.module if hasattr(model, "module") else model
    prompts = host.prompt_learner()
    features = host.text_encoder(prompts, host.tokenized_prompts)
    return F.normalize(features, dim=-1)


class FTEPPluginMixin:
    """Compile a frozen envelope after the native host model is built."""

    ftep_clip_module = None
    ftep_clip_loader = None
    ftep_canonical_templates = None

    def build_model(self):
        super().build_model()
        model = self.model.module if hasattr(self.model, "module") else self.model
        before = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        canonical = None
        if self.ftep_canonical_templates is not None:
            canonical = self.ftep_canonical_templates[self.cfg.DATASET.NAME]
        self.ftep_envelope, stats = compile_frozen_template_envelope(
            self.cfg,
            self.dm.dataset.classnames,
            self.ftep_clip_module,
            self.ftep_clip_loader,
            canonical_template=canonical,
        )
        self.ftep_envelope.to(self.device)
        after = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if before != after or next(self.ftep_envelope.parameters(), None) is not None:
            raise RuntimeError("FTEP plugin changed the host trainable parameter set")
        self.ftep_stats = stats
        print(
            f"[{self.__class__.__name__}] host={self.ftep_host_name} "
            "frozen_backbone=True added_parameters=0 added_inference_work=0 "
            "added_tunable_hyperparameters=0 "
            f"templates={stats['template_count']} "
            f"mean_radius_deg={stats['mean_radius_degrees']:.6f} "
            f"max_radius_deg={stats['max_radius_degrees']:.6f}"
        )

    def ftep_text_loss(self, text_features=None):
        if text_features is None:
            text_features = host_text_features(self.model)
        return self.ftep_envelope(text_features)

    def ftep_forward_with_text_features(self, image):
        """Run the native host once and retain its learned text features."""
        host = self.model.module if hasattr(self.model, "module") else self.model
        captured = []

        def capture_text_features(_module, _inputs, output):
            captured.append(output)

        handle = host.text_encoder.register_forward_hook(capture_text_features)
        try:
            output = self.model(image)
        finally:
            handle.remove()

        if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
            raise RuntimeError(
                "FTEP expected exactly one tensor from the host text encoder"
            )
        return output, F.normalize(captured[0], dim=-1)
