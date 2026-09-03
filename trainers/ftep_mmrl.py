"""Frozen Template Envelope Projection regularization for MMRL."""

import gc
import itertools
import math

import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from clip_mmrl import clip

from trainers.mmrl import (
    CUSTOM_TEMPLATES,
    MMRL,
    MMRL_Loss,
    load_clip_to_cpu,
)


# Method 4.1.1: dataset-level frozen template support bank (K=4).
TEMPLATE_ENVELOPES = {
    "OxfordPets": [
        "a photo of a {}, a type of pet.",
        "a close-up photo of a {} pet.",
        "a centered photo of a {} pet.",
        "a photo showing the {} pet breed.",
    ],
    "OxfordFlowers": [
        "a photo of a {}, a type of flower.",
        "a close-up photo of a {} flower.",
        "a centered photo of a {} flower.",
        "a photo showing a {} in bloom.",
    ],
    "FGVCAircraft": [
        "a photo of a {}, a type of aircraft.",
        "a side-view photo of a {} aircraft.",
        "a centered photo of a {} aircraft.",
        "a photo showing a {} aircraft.",
    ],
    "DescribableTextures": [
        "{} texture.",
        "a close-up photo of {} texture.",
        "a surface with a {} texture.",
        "a texture that appears {}.",
    ],
    "EuroSAT": [
        "a centered satellite photo of {}.",
        "a satellite image of {}.",
        "an overhead satellite photo of {}.",
        "a centered aerial image of {}.",
    ],
    "StanfordCars": [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a centered photo of a {}.",
        "a photo showing the {}.",
    ],
    "Food101": [
        "a photo of {}, a type of food.",
        "a close-up photo of {} food.",
        "a centered photo of {} food.",
        "a photo showing a dish of {}.",
    ],
    "SUN397": [
        "a photo of a {}.",
        "a wide photo of a {} scene.",
        "a centered photo of a {} scene.",
        "a photo showing a {} place.",
    ],
    "Caltech101": [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a centered photo of a {}.",
        "a photo showing the {}.",
    ],
    "UCF101": [
        "a photo of a person doing {}.",
        "a video frame of a person doing {}.",
        "a centered photo of the action {}.",
        "a photo showing someone doing {}.",
    ],
    "ImageNet": [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a centered photo of a {}.",
        "a photo showing the {}.",
    ],
}


def _sphere_log(base, points):
    """Method 4.2.1: map unit features to common tangent coordinates."""
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


class TemplateEnvelopeMMRLLoss(MMRL_Loss):
    """MMRL loss with FTEP's bounded frozen-template reference."""

    # Store frozen geometry and precompute all candidate envelope faces.
    def __init__(self, base_points, vertices, reg_weight=1.0, alpha=0.7):
        super().__init__(reg_weight=reg_weight, alpha=alpha)
        # Method 4.1.2: keep class-specific tangent base points frozen.
        self.register_buffer("envelope_base_points", base_points.float())
        # Method 4.2.2: store vertices of the bounded convex envelope.
        self.register_buffer("envelope_vertices", vertices.float())
        # Method 4.3.1: enumerate every non-empty envelope face once.
        self._faces = tuple(
            indices
            for size in range(1, vertices.shape[1] + 1)
            for indices in itertools.combinations(range(vertices.shape[1]), size)
        )

    @torch.no_grad()
    def _project_to_envelope(self, tangent):
        """Method 4.3.1: exact small-face projection onto the envelope."""
        vertices = self.envelope_vertices
        class_count = vertices.shape[0]
        best_distance = torch.full(
            (class_count,), float("inf"), device=tangent.device
        )
        best_target = torch.zeros_like(tangent)
        best_support = torch.zeros(
            class_count, device=tangent.device, dtype=torch.long
        )

        for indices in self._faces:
            # Solve the simplex-constrained least-squares problem on this face.
            face = vertices[:, indices, :]
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
                rhs_linear = torch.einsum("cmd,cd->cm", face, tangent)
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
                rhs[:, :face_size, 0] = rhs_linear
                rhs[:, face_size, 0] = 1.0
                solution = torch.linalg.solve(kkt, rhs).squeeze(-1)
                weights = solution[:, :face_size]
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
            raise RuntimeError("FTEP convex-envelope projection produced no valid face")
        return best_target, best_support

    def text_envelope_loss(self, text_features, return_support=False):
        """Method 4.3.2: penalize only displacement outside the envelope."""
        with torch.cuda.amp.autocast(enabled=False):
            text = F.normalize(text_features.float(), dim=-1)
            base = self.envelope_base_points.float()
            # Method 4.2.1: compare adapted and frozen features in one chart.
            tangent = _sphere_log(base, text)
            # Keep the nearest feasible point fixed during back-propagation.
            target, support = self._project_to_envelope(tangent.detach())
            loss = 0.5 * (tangent - target).square().sum(dim=1).mean()
        if return_support:
            return loss, support
        return loss

    # Keep MMRL classification/image terms and replace only its text prior.
    def forward(
        self,
        logits,
        logits_rep,
        image_features,
        text_features,
        image_features_clip,
        text_features_clip,
        label,
    ):
        del text_features_clip
        # Original MMRL host terms: dual classification and image-prior losses.
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)
        image_reg = 1 - F.cosine_similarity(
            image_features, image_features_clip, dim=1
        ).mean()
        # FTEP replaces MMRL's single-point text cosine regularizer.
        text_reg = self.text_envelope_loss(text_features)
        return (
            self.alpha * xe_loss1
            + (1 - self.alpha) * xe_loss2
            + self.reg_weight * image_reg
            + self.reg_weight * text_reg
        )


# Original MMRL process inherited unchanged: check_cfg, training, testing, and loading.
@TRAINER_REGISTRY.register()
class FTEP_MMRL(MMRL):
    """MMRL host with bounded geodesic template-envelope preservation."""

    @torch.no_grad()
    def _compile_template_envelopes(self, classnames):
        """Compile the frozen bank, tangent bases, and bounded envelopes."""
        dataset = self.cfg.DATASET.NAME
        # Method 4.1.1: retain the native MMRL template as the first support point.
        templates = TEMPLATE_ENVELOPES.get(
            dataset, [CUSTOM_TEMPLATES[dataset]]
        )
        if len(templates) < 2:
            raise RuntimeError("FTEP requires at least two equivalent templates")
        if templates[0] != CUSTOM_TEMPLATES[dataset]:
            raise RuntimeError("FTEP envelope must include the native MMRL template first")

        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        encoder_device = next(self.text_encoder_clip.parameters()).device
        token_source = None
        try:
            token_source = load_clip_to_cpu(self.cfg)
            self.text_encoder_clip.to(self.device)
            self.text_encoder_clip.eval()
            # Frozen CLIP encodes the support bank without image supervision.
            features = []
            for template in templates:
                tokenized = torch.cat([
                    clip.tokenize(template.format(name.replace("_", " ")))
                    for name in classnames
                ])
                embeddings = token_source.token_embedding(tokenized).type(
                    self.dtype
                )
                encoded = self.text_encoder_clip(
                    embeddings.to(self.device), tokenized.to(self.device)
                )
                features.append(F.normalize(encoded.float(), dim=-1))
        finally:
            self.text_encoder_clip.to(encoder_device)
            del token_source
            gc.collect()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        bank = torch.stack(features, dim=1)
        # Method 4.1.2: class-specific normalized spherical mean.
        base_points = F.normalize(bank.mean(dim=1), dim=-1)
        # Method 4.2.1/4.2.2: common tangent coordinates form the convex envelope.
        vertices = _sphere_log(base_points.unsqueeze(1), bank)
        radii = vertices.norm(dim=-1)
        canonical_error = (
            bank[:, 0] - self.text_features_clip.float()
        ).abs().max().item()
        stats = {
            "template_count": len(templates),
            "mean_radius_degrees": float(radii.mean().item() * 180.0 / math.pi),
            "max_radius_degrees": float(radii.max().item() * 180.0 / math.pi),
            "canonical_error": float(canonical_error),
        }
        return base_points, vertices, stats

    @torch.no_grad()
    def _initial_regularizer_diagnostics(self, criterion):
        """Compare native MMRL point guidance with the initial FTEP envelope loss."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        text_tokens, _ = model.representation_learner()
        text = model.text_encoder(
            model.prompt_embeddings, model.tokenized_prompts, text_tokens
        )
        text = F.normalize(text.float(), dim=-1)
        point_loss = 1 - F.cosine_similarity(
            text, self.text_features_clip.float(), dim=1
        ).mean()
        envelope_loss, support = criterion.text_envelope_loss(
            text, return_support=True
        )
        return (
            float(point_loss.item()),
            float(envelope_loss.item()),
            float(support.float().mean().item()),
            int(support.max().item()),
        )

    def build_model(self):
        # Original MMRL host setup: build the frozen CLIP and trainable adapters.
        super().build_model()
        classnames = self.dm.dataset.classnames
        # FTEP additions: compile frozen support and replace the text criterion.
        base_points, vertices, stats = self._compile_template_envelopes(classnames)
        criterion = TemplateEnvelopeMMRLLoss(
            base_points,
            vertices,
            reg_weight=self.cfg.TRAINER.MMRL.REG_WEIGHT,
            alpha=self.cfg.TRAINER.MMRL.ALPHA,
        ).to(self.device)
        point_loss, envelope_loss, mean_support, max_support = (
            self._initial_regularizer_diagnostics(criterion)
        )
        self.criterion = criterion

        # Preserve MMRL's trainable parameter set; FTEP adds no trainable parameters.
        model = self.model.module if hasattr(self.model, "module") else self.model
        allowed = ("representation_learner", "image_encoder.proj_rep")
        invalid_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not any(item in name for item in allowed)
        ]
        if stats["canonical_error"] > 1e-5 or invalid_trainable:
            raise RuntimeError(
                "FTEP integrity check failed: "
                f"canonical_error={stats['canonical_error']:.8f}, "
                f"invalid_trainable={invalid_trainable}"
            )

        self.ftep_stats = stats
        print(
            "[FTEP_MMRL] frozen_backbone=True added_parameters=0 "
            "added_inference_work=0 added_tunable_hyperparameters=0 "
            f"templates={stats['template_count']} "
            f"mean_radius_deg={stats['mean_radius_degrees']:.6f} "
            f"max_radius_deg={stats['max_radius_degrees']:.6f} "
            f"canonical_error={stats['canonical_error']:.8f} "
            f"initial_point_loss={point_loss:.8f} "
            f"initial_envelope_loss={envelope_loss:.8f} "
            f"initial_mean_support={mean_support:.4f} "
            f"initial_max_support={max_support}"
        )
