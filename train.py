"""Minimal Dassl entry point for CoOp, DePT, MMRL and FTEP_MMRL.

The CLIP backbone remains frozen by each trainer. Dataset roots and model
weights are supplied at runtime; no paths or credentials are hard-coded.
"""

import argparse

import torch
from yacs.config import CfgNode as CN

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

# Register only the supported base-to-new datasets.
import datasets.caltech101  # noqa: F401
import datasets.dtd  # noqa: F401
import datasets.eurosat  # noqa: F401
import datasets.fgvc_aircraft  # noqa: F401
import datasets.food101  # noqa: F401
import datasets.imagenet  # noqa: F401
import datasets.oxford_flowers  # noqa: F401
import datasets.oxford_pets  # noqa: F401
import datasets.stanford_cars  # noqa: F401
import datasets.sun397  # noqa: F401
import datasets.ucf101  # noqa: F401

# Import side effects register the four trainers in TRAINER_REGISTRY.
import trainers.coop  # noqa: F401
import trainers.dept_coop  # noqa: F401
import trainers.mmrl  # noqa: F401
import trainers.ftep_mmrl  # noqa: F401


def print_args(args, cfg):
    print("***************\n** Arguments **\n***************")
    for key in sorted(vars(args)):
        print(f"{key}: {getattr(args, key)}")
    print("************\n** Config **\n************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """Add only configuration nodes consumed by the retained trainers."""
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TASK = "B2N"

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.MMRL = CN()
    cfg.TRAINER.MMRL.ALPHA = 0.7
    cfg.TRAINER.MMRL.REG_WEIGHT = 1.0
    cfg.TRAINER.MMRL.N_REP_TOKENS = 5
    cfg.TRAINER.MMRL.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.TRAINER.MMRL.PREC = "amp"
    cfg.TRAINER.MMRL.REP_DIM = 512

    # Official CoOp+DePT settings.
    cfg.TRAINER.LINEAR_PROBE = CN()
    cfg.TRAINER.LINEAR_PROBE.TYPE = "linear"
    cfg.TRAINER.LINEAR_PROBE.CLS_WEIGHT = 0.7
    cfg.TRAINER.LINEAR_PROBE.WEIGHT = 0.7
    cfg.TRAINER.LINEAR_PROBE.TEST_TIME_FUSION = True
    cfg.TRAINER.FILM = CN()
    cfg.TRAINER.FILM.LINEAR_PROBE = True
    cfg.TRAINER.NAMES_TO_UPDATE = ["prompt_learner", "linear_probe", "film"]

    # DPC CONFIGS
    cfg.SPLE = CN()
    # Fine-tuning ckeckpoint and epoch of pre-tuned backbone (e.g., CoOp). Must be provided before tuning on DPC.
    cfg.SPLE.BACK_CKPT_PATH = "../output/caltech101-N3-Final/CoOp/vit_b16_ep100_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-100"
    cfg.SPLE.BACK_CKPT_EPOCH = 100
    cfg.SPLE.INFER_TOPK = 8  # Top-K for dynamic hard negative sampler
    cfg.SPLE.INFER_NONREPEAT = True  # No repeat object in a mini-batch
    cfg.SPLE.PIC_LIB = "../DATA/SPLE_database/SPLE_Caltech101.json"
    cfg.SPLE.LOSS_KD_WEIGHT = 0.5  # [DPC_PromptKD] Weight for fusing DPC_loss (L_cl) and PromptKD_loss (L_kd)
    cfg.SPLE.KD_INFER = ""  # [DPC_PromptKD] Name of inference head
    cfg.SPLE.SPLE_TRAINER = CN()
    cfg.SPLE.SPLE_TRAINER.SPLE_INIT = True  # [DPC] Use DPC to init
    cfg.SPLE.SPLE_TRAINER.SPLE_BASE_TRAIN = True  # [DPC] Use same base classes (keep 'True' to avoid data leakage)
    cfg.SPLE.STACK = CN()
    cfg.SPLE.STACK.MODE = "void"  # [DISCARD] DO NOT CHANGE
    cfg.SPLE.STACK.WEIGHT = 0.5  # [DPC] Weight for base
    cfg.SPLE.STACK.WEIGHT_FOR_NEW = 0.2  # [DPC] Weight for new
    cfg.SPLE.STACK.LOOP_DEPTH = 0  # [DISCARD] DO NOT CHANGE


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_cfg(cfg, args)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)
    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="", help="dataset root")
    parser.add_argument("--output-dir", default="", help="output directory")
    parser.add_argument("--resume", default="", help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config-file", default="")
    parser.add_argument("--dataset-config-file", default="")
    parser.add_argument("--trainer", default="")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    main(parser.parse_args())
