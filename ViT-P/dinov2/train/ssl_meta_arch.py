# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk

import dinov2.utils.utils as dinov2_utils

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()


        student_backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)

        dinov2_utils.load_pretrained_weights(student_backbone, cfg.student.pretrained_weights, "teacher")
        
        student_model_dict["backbone"] = student_backbone

        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        # if cfg.student.pretrained_weights:
        #     chkpt = torch.load(cfg.student.pretrained_weights)
        #     logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
        #     student_backbone.load_state_dict(chkpt, strict=False)
        
        
        self.embed_dim = embed_dim





        dino_head = partial(
            DINOHead,
        )
        self.dino_loss = nn.CrossEntropyLoss()



        student_model_dict["dino_head"] = dino_head()




        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        
        for p in self.student["dino_head"].parameters():
            p.requires_grad = False

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, data):

        loss_dict = {}
        logits = self.student.backbone(data['image'].cuda(non_blocking=True), data['points'].cuda(non_blocking=True))
        
        
        outputs = self.student.dino_head(logits)


        loss = self.dino_loss(
            outputs,
            data['label'].cuda(non_blocking=True).view(-1).long(),
        ) 


            # store for display
        loss_dict["dino_local_crops_loss"] = loss



        self.backprop_loss(loss)

        # reshard_fsdp_model(self.student)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # self.student.dino_head._streams = (
            #     self.teacher.dino_head._streams
            # ) = self.student.backbone._streams = self.teacher.backbone._streams
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.student.backbone, attr)  
                setattr(self.student.dino_head, attr, stream) 
                setattr(self.student.backbone, attr, stream) 
            self.need_to_synchronize_fsdp_streams = False

        
    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()

    def eval(self):
        self.student.eval()

        
    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            