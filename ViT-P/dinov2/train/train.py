# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.utils.scheduler import WarmupCosineSchedule
from dinov2.train.ssl_meta_arch import SSLMetaArch

from dinov2.eval.metrics import MetricType, build_metric

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("ViT-P training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


@torch.inference_mode()
def do_test(cfg, model, data_loader, iteration):
    model.eval()
    logger.info("running validation !")

    num_classes = cfg.student.num_classes
    metric = build_metric(MetricType.MEAN_ACCURACY, num_classes=num_classes)
    metric = metric.cuda()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    
    for data in metric_logger.log_every(data_loader, 100, header):
        
        logits = model.student.backbone(data['image'].cuda(non_blocking=True), data['points'].cuda(non_blocking=True))
        outputs = model.student.dino_head(logits)
        targets = data['label'].cuda(non_blocking=True)

        metric_inputs = {
            "preds": outputs.reshape(-1,num_classes),
            "target": targets.reshape(-1),
        }
        metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    
    results_dict_temp = metric.compute()
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    logger.info("")
    
    max_accuracy = 0
    top1_accuracy = results_dict_temp["top-1"]
    top5_accuracy = results_dict_temp["top-5"]
    
    if top1_accuracy > max_accuracy:
        max_accuracy = top1_accuracy
        
    logger.info(f"Top_1_accuracy: {top1_accuracy}")
    logger.info(f"Top_5_accuracy: {top5_accuracy}")
    # logger.info(f"max_accuracy: {max_accuracy}")

    new_state_dict = model.student["backbone"].state_dict()

    if distributed.is_main_process():
        # iterstring = str(iteration)
        # eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        # os.makedirs(eval_dir, exist_ok=True)
        # # save teacher checkpoint
        # ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        ckp_path =  "./checkpoint.pth"
        torch.save(new_state_dict, ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    # optimizer = build_optimizer(cfg, model.get_params_groups())
    

    named_parameters = list(model.student["backbone"].named_parameters())
    # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad and n != 'cls_embeddings.weight']
    rest_params = [p for n, p in named_parameters if p.requires_grad and n != '_fsdp_wrapped_module.cls_embeddings.weight']
    cls_embd = [p for n, p in named_parameters if p.requires_grad and n == '_fsdp_wrapped_module.cls_embeddings.weight']

    optimizer = torch.optim.SGD([   
                        {"params": cls_embd, 'lr': 1e-2},
                        {"params": rest_params},
                        ],
                        lr=cfg.optim.lr,
                        momentum=0.9,
                        weight_decay = cfg.optim.weight_decay)


    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=(cfg.optim.warmup_epochs * cfg.train.OFFICIAL_EPOCH_LENGTH), t_total=(cfg.optim["epochs"] * cfg.train.OFFICIAL_EPOCH_LENGTH))   

    # optimizer.load_state_dict(torch.load("./optimizer.pth"))
    # scheduler.load_state_dict(torch.load("./scheduler.pth"))
    # (
    #     lr_schedule,
    #     wd_schedule,
    #     momentum_schedule,
    #     teacher_temp_schedule,
    #     last_layer_lr_schedule,
    # ) = build_schedulers(cfg)

    # checkpointer
    # checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    # start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    start_iter = 0
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    # periodic_checkpointer = PeriodicCheckpointer(
    #     checkpointer,
    #     period=3 * OFFICIAL_EPOCH_LENGTH,
    #     max_iter=max_iter,
    #     max_to_keep=3,
    # )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size


    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        split="train",
        image_size=img_size,
        n_points=cfg.student.num_points,
    )

    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=None,
        persistent_workers=False,
    )


    val_dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        split="val",
        image_size=img_size,
        n_points=cfg.student.num_points,
    )

    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.DISTRIBUTED
    val_data_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        sampler_type=sampler_type,
        drop_last=False,
        collate_fn=None,
        persistent_workers=False,
    )


    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    
    for data in metric_logger.log_every(
        data_loader,
        100,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data['image'].shape[0] 
        if iteration > max_iter:
            return
            

        # apply schedules

        # lr = lr_schedule[iteration]
        # wd = wd_schedule[iteration]
        # mom = momentum_schedule[iteration]
        # last_layer_lr = last_layer_lr_schedule[iteration]
        # apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute lossess

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()
        scheduler.step()
        # perform teacher EMA update


        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, val_data_loader, f"training_{iteration}")
            if distributed.is_main_process():
                torch.save(optimizer.state_dict(), "./optimizer.pth")
                torch.save(scheduler.state_dict(), "./scheduler.pth")
            torch.cuda.synchronize()
            model.train()
        # periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    torch.distributed.destroy_process_group()
    return 0


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, val_data_loader, f"training_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
