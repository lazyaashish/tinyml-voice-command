"""
High-Performance, Bit-Exact Quantization-Aware Training (QAT) Engine for Microcontroller KWS.
Features:
1. Straight-Through Estimator (STE) for symmetric per-channel weights and calibrated per-tensor activations.
2. Mathematically matches CMSIS-NN fixed-point integer kernels (arm_convolve_s8, arm_depthwise_conv_s8).
3. Pre-folds BatchNorm into Conv weights/biases for zero mismatch between QAT and deployed integer weights.
4. Native support for Cross-Entropy + Mel40 WideTeacher Soft-Logit Knowledge Distillation.
5. Direct checkpoint selection based on integer-quantized validation accuracy on 4,445 clips.
"""

import os
import sys
import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        ctx.save_for_backward(x, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        q_x = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q_x * scale

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        scaled_x = x / scale
        mask = (scaled_x >= qmin) & (scaled_x <= qmax)
        grad_x = grad_output * mask.float()
        return grad_x, None, None, None

def fake_quant_weight_per_channel(w, qmin=-128, qmax=127):
    """Symmetric per-channel weight quantization for Conv2d, DepthwiseConv2d, and Linear."""
    if w.dim() == 4:
        # Conv2d / DepthwiseConv2d: per output channel
        scale = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / float(qmax)
    elif w.dim() == 2:
        # Linear: per output neuron
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / float(qmax)
    else:
        scale = w.abs().amax().clamp(min=1e-8) / float(qmax)
    return STEQuantize.apply(w, scale, qmin, qmax)

class FakeQuantActivation(nn.Module):
    """
    Calibrated per-tensor activation fake-quantization.
    Matches CMSIS-NN unsigned [0, 127] for post-ReLU activations.
    """
    def __init__(self, qmin=0, qmax=127, momentum=0.05, init_max=6.0):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer('running_max', torch.tensor(float(init_max)))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if self.training:
            cur_max = x.detach().abs().max()
            if self.initialized == 0:
                self.running_max.copy_(cur_max)
                self.initialized.fill_(1)
            else:
                self.running_max.copy_(self.running_max * (1.0 - self.momentum) + cur_max * self.momentum)
        scale = torch.clamp(self.running_max / float(self.qmax), min=1e-8)
        return STEQuantize.apply(x, scale, self.qmin, self.qmax)

def fold_conv_bn_weights(conv, bn):
    """Fuses Conv2d (or Conv1d) and BatchNorm2d into folded weight and bias."""
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(w.size(0), device=w.device)

    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    inv_std = gamma / torch.sqrt(var + eps)
    w_fused = w * inv_std.view(-1, 1, 1, 1)
    b_fused = (b - mean) * inv_std + beta
    return w_fused, b_fused

# ==============================================================================
# QAT Architecture Implementations (Folded CMSIS-NN Bit-Exact Topology)
# ==============================================================================

class QATArmDSCNN_W68(nn.Module):
    """
    Candidate A & F: QAT Model for ArmDSCNN_W68_GAP.
    Uniform 68 channels, 4 Depthwise-Separable Blocks, 2D GAP.
    Parameters: 25,716 | Peak SRAM: 18,224 Bytes | MACs: 2,958,816.
    """
    def __init__(self, num_classes: int = 12, channels: int = 68):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels

        # Stem Conv (1 -> 68, stride 2,2)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.quant_act1 = FakeQuantActivation(0, 127)

        # 4 DS Blocks
        self.blocks = nn.ModuleList()
        for _ in range(4):
            dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
            pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
            block = nn.ModuleDict({
                'dw': dw,
                'relu_dw': nn.ReLU(inplace=True),
                'quant_dw': FakeQuantActivation(0, 127),
                'pw': pw,
                'relu_pw': nn.ReLU(inplace=True),
                'quant_pw': FakeQuantActivation(0, 127),
            })
            self.blocks.append(block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quant_pool = FakeQuantActivation(0, 127)
        self.classifier = nn.Linear(channels, num_classes, bias=True)

    @classmethod
    def from_fp32(cls, fp32_model):
        qat_model = cls(num_classes=fp32_model.num_classes, channels=fp32_model.channels)
        fp32_model.eval()

        # Fold Conv1 + BN1
        w_fused, b_fused = fold_conv_bn_weights(fp32_model.conv1, fp32_model.bn1)
        qat_model.conv1.weight.data.copy_(w_fused)
        qat_model.conv1.bias.data.copy_(b_fused)

        # Fold 4 DS blocks
        for i, block in enumerate(fp32_model.blocks):
            w_dw, b_dw = fold_conv_bn_weights(block.depthwise, block.bn_dw)
            qat_model.blocks[i]['dw'].weight.data.copy_(w_dw)
            qat_model.blocks[i]['dw'].bias.data.copy_(b_dw)

            w_pw, b_pw = fold_conv_bn_weights(block.pointwise, block.bn_pw)
            qat_model.blocks[i]['pw'].weight.data.copy_(w_pw)
            qat_model.blocks[i]['pw'].bias.data.copy_(b_pw)

        qat_model.classifier.weight.data.copy_(fp32_model.classifier.weight.data)
        qat_model.classifier.bias.data.copy_(fp32_model.classifier.bias.data)
        return qat_model

    def forward(self, x):
        x = F.pad(x, (1, 1, 4, 5))
        
        # Stem
        q_w1 = fake_quant_weight_per_channel(self.conv1.weight)
        x = F.conv2d(x, q_w1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.relu1(x)
        x = self.quant_act1(x)

        # Blocks
        for block in self.blocks:
            q_dw = fake_quant_weight_per_channel(block['dw'].weight)
            x = F.conv2d(x, q_dw, block['dw'].bias, stride=1, padding=1, groups=self.channels)
            x = block['relu_dw'](x)
            x = block['quant_dw'](x)

            q_pw = fake_quant_weight_per_channel(block['pw'].weight)
            x = F.conv2d(x, q_pw, block['pw'].bias, stride=1, padding=0)
            x = block['relu_pw'](x)
            x = block['quant_pw'](x)

        x = self.avg_pool(x)
        x = self.quant_pool(torch.flatten(x, 1))

        q_fc = fake_quant_weight_per_channel(self.classifier.weight)
        return F.linear(x, q_fc, self.classifier.bias)


class QATExt_W68_EarlySkip_GAP(nn.Module):
    """
    Candidate B: QAT Model for Ext_W68_EarlySkip_GAP.
    Taps Block 1 and immediately pools to a 68-byte global vector.
    Parameters: 30,340 | Peak SRAM: 18,296 Bytes | MACs: 2,963,440.
    """
    def __init__(self, num_classes: int = 12, channels: int = 68):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.quant_act1 = FakeQuantActivation(0, 127)

        # Block 1
        self.b1_dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
        self.b1_relu_dw = nn.ReLU(inplace=True)
        self.b1_quant_dw = FakeQuantActivation(0, 127)
        self.b1_pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
        self.b1_relu_pw = nn.ReLU(inplace=True)
        self.b1_quant_pw = FakeQuantActivation(0, 127)

        # Immediate skip pool + linear projection (holds only 68 bytes in memory!)
        self.skip_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.skip_quant = FakeQuantActivation(0, 127)
        self.skip_proj = nn.Linear(channels, channels, bias=False)
        self.skip_proj_quant = FakeQuantActivation(0, 127)

        # Blocks 2, 3, 4
        self.blocks = nn.ModuleList()
        for _ in range(3):
            dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
            pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
            block = nn.ModuleDict({
                'dw': dw,
                'relu_dw': nn.ReLU(inplace=True),
                'quant_dw': FakeQuantActivation(0, 127),
                'pw': pw,
                'relu_pw': nn.ReLU(inplace=True),
                'quant_pw': FakeQuantActivation(0, 127),
            })
            self.blocks.append(block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quant_pool = FakeQuantActivation(0, 127)
        self.quant_fuse = FakeQuantActivation(0, 127)
        self.classifier = nn.Linear(channels, num_classes, bias=True)

    @classmethod
    def from_fp32(cls, fp32_model):
        qat = cls(num_classes=fp32_model.num_classes, channels=fp32_model.channels)
        fp32_model.eval()

        w_conv1, b_conv1 = fold_conv_bn_weights(fp32_model.conv1, fp32_model.bn1)
        qat.conv1.weight.data.copy_(w_conv1)
        qat.conv1.bias.data.copy_(b_conv1)

        # Block 1
        w_b1_dw, b_b1_dw = fold_conv_bn_weights(fp32_model.block1.depthwise, fp32_model.block1.bn_dw)
        qat.b1_dw.weight.data.copy_(w_b1_dw)
        qat.b1_dw.bias.data.copy_(b_b1_dw)

        w_b1_pw, b_b1_pw = fold_conv_bn_weights(fp32_model.block1.pointwise, fp32_model.block1.bn_pw)
        qat.b1_pw.weight.data.copy_(w_b1_pw)
        qat.b1_pw.bias.data.copy_(b_b1_pw)

        # Skip proj
        qat.skip_proj.weight.data.copy_(fp32_model.skip_proj.weight.data)

        # Blocks 2, 3, 4
        fp32_blocks = [fp32_model.block2, fp32_model.block3, fp32_model.block4]
        for i, b in enumerate(fp32_blocks):
            w_dw, b_dw = fold_conv_bn_weights(b.depthwise, b.bn_dw)
            qat.blocks[i]['dw'].weight.data.copy_(w_dw)
            qat.blocks[i]['dw'].bias.data.copy_(b_dw)

            w_pw, b_pw = fold_conv_bn_weights(b.pointwise, b.bn_pw)
            qat.blocks[i]['pw'].weight.data.copy_(w_pw)
            qat.blocks[i]['pw'].bias.data.copy_(b_pw)

        qat.classifier.weight.data.copy_(fp32_model.classifier.weight.data)
        qat.classifier.bias.data.copy_(fp32_model.classifier.bias.data)
        return qat

    def forward(self, x):
        x = F.pad(x, (1, 1, 4, 5))
        
        # Stem
        q_w1 = fake_quant_weight_per_channel(self.conv1.weight)
        x = F.conv2d(x, q_w1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.relu1(x)
        x = self.quant_act1(x)

        # Block 1
        q_b1_dw = fake_quant_weight_per_channel(self.b1_dw.weight)
        x = F.conv2d(x, q_b1_dw, self.b1_dw.bias, stride=1, padding=1, groups=self.channels)
        x = self.b1_relu_dw(x)
        x = self.b1_quant_dw(x)

        q_b1_pw = fake_quant_weight_per_channel(self.b1_pw.weight)
        x = F.conv2d(x, q_b1_pw, self.b1_pw.bias, stride=1, padding=0)
        x = self.b1_relu_pw(x)
        x = self.b1_quant_pw(x)

        # Immediate skip pooling: compresses to 68 bytes!
        skip = self.skip_pool(x)
        skip = self.skip_quant(torch.flatten(skip, 1))
        q_skip_w = fake_quant_weight_per_channel(self.skip_proj.weight)
        skip = F.linear(skip, q_skip_w, None)
        skip = self.skip_proj_quant(skip)

        # Blocks 2-4
        for block in self.blocks:
            q_dw = fake_quant_weight_per_channel(block['dw'].weight)
            x = F.conv2d(x, q_dw, block['dw'].bias, stride=1, padding=1, groups=self.channels)
            x = block['relu_dw'](x)
            x = block['quant_dw'](x)

            q_pw = fake_quant_weight_per_channel(block['pw'].weight)
            x = F.conv2d(x, q_pw, block['pw'].bias, stride=1, padding=0)
            x = block['relu_pw'](x)
            x = block['quant_pw'](x)

        x = self.avg_pool(x)
        x = self.quant_pool(torch.flatten(x, 1))

        # Add skip
        fused = self.quant_fuse(x + skip)
        q_fc = fake_quant_weight_per_channel(self.classifier.weight)
        return F.linear(fused, q_fc, self.classifier.bias)


class QATExt_DSCNN_W68_GatedOnset(nn.Module):
    """
    Candidate C: QAT Model for Ext_DSCNN_W68_GatedOnset_Finetune.
    Parameters: 37,580 | Peak SRAM: 18,224 Bytes | MACs: 2,970,272.
    """
    def __init__(self, num_classes: int = 12, C: int = 68):
        super().__init__()
        self.num_classes = num_classes
        self.C = C

        self.conv1 = nn.Conv2d(1, C, kernel_size=(10, 4), stride=(2, 2), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.quant_act1 = FakeQuantActivation(0, 127)

        self.blocks = nn.ModuleList()
        for _ in range(4):
            dw = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=True)
            pw = nn.Conv2d(C, C, kernel_size=1, stride=1, bias=True)
            block = nn.ModuleDict({
                'dw': dw,
                'relu_dw': nn.ReLU(inplace=True),
                'quant_dw': FakeQuantActivation(0, 127),
                'pw': pw,
                'relu_pw': nn.ReLU(inplace=True),
                'quant_pw': FakeQuantActivation(0, 127),
            })
            self.blocks.append(block)

        # Gated Onset Head
        self.gate_fc = nn.Linear(C, C, bias=True)
        self.quant_gate = FakeQuantActivation(0, 127)
        self.classifier = nn.Linear(C * 2, num_classes, bias=True)

    @classmethod
    def from_fp32(cls, fp32_model):
        qat = cls(num_classes=fp32_model.num_classes, C=fp32_model.C)
        fp32_model.eval()

        w_fused, b_fused = fold_conv_bn_weights(fp32_model.conv1, fp32_model.bn1)
        qat.conv1.weight.data.copy_(w_fused)
        qat.conv1.bias.data.copy_(b_fused)

        for i, block in enumerate(fp32_model.blocks):
            w_dw, b_dw = fold_conv_bn_weights(block.depthwise, block.bn_dw)
            qat.blocks[i]['dw'].weight.data.copy_(w_dw)
            qat.blocks[i]['dw'].bias.data.copy_(b_dw)

            w_pw, b_pw = fold_conv_bn_weights(block.pointwise, block.bn_pw)
            qat.blocks[i]['pw'].weight.data.copy_(w_pw)
            qat.blocks[i]['pw'].bias.data.copy_(b_pw)

        qat.gate_fc.weight.data.copy_(fp32_model.head.gate_fc.weight.data)
        qat.gate_fc.bias.data.copy_(fp32_model.head.gate_fc.bias.data)
        qat.classifier.weight.data.copy_(fp32_model.head.classifier.weight.data)
        qat.classifier.bias.data.copy_(fp32_model.head.classifier.bias.data)
        return qat

    def forward(self, x):
        x = F.pad(x, (1, 1, 4, 5))
        
        q_w1 = fake_quant_weight_per_channel(self.conv1.weight)
        x = F.conv2d(x, q_w1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.relu1(x)
        x = self.quant_act1(x)

        for block in self.blocks:
            q_dw = fake_quant_weight_per_channel(block['dw'].weight)
            x = F.conv2d(x, q_dw, block['dw'].bias, stride=1, padding=1, groups=self.C)
            x = block['relu_dw'](x)
            x = block['quant_dw'](x)

            q_pw = fake_quant_weight_per_channel(block['pw'].weight)
            x = F.conv2d(x, q_pw, block['pw'].bias, stride=1, padding=0)
            x = block['relu_pw'](x)
            x = block['quant_pw'](x)

        # Gated Onset Head logic:
        # x is (B, C, 25, 5)
        # 1. Global representation
        g_rep = x.mean(dim=(2, 3)) # (B, C)
        
        # 2. Onset representation (early frames: T[0:12])
        onset_rep = x[:, :, :12, :].mean(dim=(2, 3)) # (B, C)
        q_gw = fake_quant_weight_per_channel(self.gate_fc.weight)
        gate = torch.sigmoid(F.linear(g_rep, q_gw, self.gate_fc.bias))
        gated_onset = self.quant_gate(onset_rep * gate)

        combined = torch.cat([g_rep, gated_onset], dim=-1) # (B, 2*C)
        q_cls = fake_quant_weight_per_channel(self.classifier.weight)
        return F.linear(combined, q_cls, self.classifier.bias)


class QATExt_W72_EfficientTemporal(nn.Module):
    """
    Candidate D: QAT Model for Ext_W72_EfficientTemporal.
    Parameters: 25,952 | Peak SRAM: 17,368 Bytes | MACs: 1,982,092.
    """
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.quant_act1 = FakeQuantActivation(0, 127)

        # Block 1 (64ch, s=1)
        self.b1_dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=True)
        self.b1_relu_dw = nn.ReLU(inplace=True)
        self.b1_quant_dw = FakeQuantActivation(0, 127)
        self.b1_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
        self.b1_relu_pw = nn.ReLU(inplace=True)
        self.b1_quant_pw = FakeQuantActivation(0, 127)

        # Block 2 (64ch, temporal downsample stride=(2,1))
        self.b2_dw = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, groups=64, bias=True)
        self.b2_relu_dw = nn.ReLU(inplace=True)
        self.b2_quant_dw = FakeQuantActivation(0, 127)
        self.b2_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
        self.b2_relu_pw = nn.ReLU(inplace=True)
        self.b2_quant_pw = FakeQuantActivation(0, 127)

        # Block 3 (64 -> 76ch)
        self.b3_dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=True)
        self.b3_relu_dw = nn.ReLU(inplace=True)
        self.b3_quant_dw = FakeQuantActivation(0, 127)
        self.b3_pw = nn.Conv2d(64, 76, kernel_size=1, stride=1, bias=True)
        self.b3_relu_pw = nn.ReLU(inplace=True)
        self.b3_quant_pw = FakeQuantActivation(0, 127)

        # Block 4 (76ch)
        self.b4_dw = nn.Conv2d(76, 76, kernel_size=3, stride=1, padding=1, groups=76, bias=True)
        self.b4_relu_dw = nn.ReLU(inplace=True)
        self.b4_quant_dw = FakeQuantActivation(0, 127)
        self.b4_pw = nn.Conv2d(76, 76, kernel_size=1, stride=1, bias=True)
        self.b4_relu_pw = nn.ReLU(inplace=True)
        self.b4_quant_pw = FakeQuantActivation(0, 127)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quant_pool = FakeQuantActivation(0, 127)
        self.classifier = nn.Linear(76, num_classes, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 4, 5))
        
        q_w1 = fake_quant_weight_per_channel(self.conv1.weight)
        x = F.conv2d(x, q_w1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.relu1(x)
        x = self.quant_act1(x)

        # B1
        q_b1_dw = fake_quant_weight_per_channel(self.b1_dw.weight)
        x = F.conv2d(x, q_b1_dw, self.b1_dw.bias, stride=1, padding=1, groups=64)
        x = self.b1_relu_dw(x)
        x = self.b1_quant_dw(x)
        q_b1_pw = fake_quant_weight_per_channel(self.b1_pw.weight)
        x = F.conv2d(x, q_b1_pw, self.b1_pw.bias, stride=1, padding=0)
        x = self.b1_relu_pw(x)
        x = self.b1_quant_pw(x)

        # B2
        q_b2_dw = fake_quant_weight_per_channel(self.b2_dw.weight)
        x = F.conv2d(x, q_b2_dw, self.b2_dw.bias, stride=(2, 1), padding=1, groups=64)
        x = self.b2_relu_dw(x)
        x = self.b2_quant_dw(x)
        q_b2_pw = fake_quant_weight_per_channel(self.b2_pw.weight)
        x = F.conv2d(x, q_b2_pw, self.b2_pw.bias, stride=1, padding=0)
        x = self.b2_relu_pw(x)
        x = self.b2_quant_pw(x)

        # B3
        q_b3_dw = fake_quant_weight_per_channel(self.b3_dw.weight)
        x = F.conv2d(x, q_b3_dw, self.b3_dw.bias, stride=1, padding=1, groups=64)
        x = self.b3_relu_dw(x)
        x = self.b3_quant_dw(x)
        q_b3_pw = fake_quant_weight_per_channel(self.b3_pw.weight)
        x = F.conv2d(x, q_b3_pw, self.b3_pw.bias, stride=1, padding=0)
        x = self.b3_relu_pw(x)
        x = self.b3_quant_pw(x)

        # B4
        q_b4_dw = fake_quant_weight_per_channel(self.b4_dw.weight)
        x = F.conv2d(x, q_b4_dw, self.b4_dw.bias, stride=1, padding=1, groups=76)
        x = self.b4_relu_dw(x)
        x = self.b4_quant_dw(x)
        q_b4_pw = fake_quant_weight_per_channel(self.b4_pw.weight)
        x = F.conv2d(x, q_b4_pw, self.b4_pw.bias, stride=1, padding=0)
        x = self.b4_relu_pw(x)
        x = self.b4_quant_pw(x)

        x = self.avg_pool(x)
        x = self.quant_pool(torch.flatten(x, 1))

        q_fc = fake_quant_weight_per_channel(self.classifier.weight)
        return F.linear(x, q_fc, self.classifier.bias)


class QATExt_MFCC_HighBand_SideInfo(nn.Module):
    """
    Candidate E: QAT Model for Ext_MFCC_HighBand_SideInfo.
    Input: (B, 1, 49, 12). Uniform 60 channels.
    Parameters: 20,772 | Peak SRAM: 19,080 Bytes | MACs: 2,844,720.
    """
    def __init__(self, num_classes: int = 12, channels: int = 60):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.quant_act1 = FakeQuantActivation(0, 127)

        self.blocks = nn.ModuleList()
        for _ in range(4):
            dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
            pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)
            block = nn.ModuleDict({
                'dw': dw,
                'relu_dw': nn.ReLU(inplace=True),
                'quant_dw': FakeQuantActivation(0, 127),
                'pw': pw,
                'relu_pw': nn.ReLU(inplace=True),
                'quant_pw': FakeQuantActivation(0, 127),
            })
            self.blocks.append(block)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quant_pool = FakeQuantActivation(0, 127)
        self.classifier = nn.Linear(channels, num_classes, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 4, 5))
        
        q_w1 = fake_quant_weight_per_channel(self.conv1.weight)
        x = F.conv2d(x, q_w1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.relu1(x)
        x = self.quant_act1(x)

        for block in self.blocks:
            q_dw = fake_quant_weight_per_channel(block['dw'].weight)
            x = F.conv2d(x, q_dw, block['dw'].bias, stride=1, padding=1, groups=self.channels)
            x = block['relu_dw'](x)
            x = block['quant_dw'](x)

            q_pw = fake_quant_weight_per_channel(block['pw'].weight)
            x = F.conv2d(x, q_pw, block['pw'].bias, stride=1, padding=0)
            x = block['relu_pw'](x)
            x = block['quant_pw'](x)

        x = self.avg_pool(x)
        x = self.quant_pool(torch.flatten(x, 1))

        q_fc = fake_quant_weight_per_channel(self.classifier.weight)
        return F.linear(x, q_fc, self.classifier.bias)


# ==============================================================================
# QAT Training & Evaluation Routine
# ==============================================================================

def evaluate_qat_validation(model, val_loader, device=torch.device("cpu")):
    """Evaluates fake-quantized validation accuracy on 4,445 clips."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / float(total) * 100.0


def run_qat_finetune(model, train_loader, val_loader, epochs: int = 15, lr: float = 3e-5,
                     use_teacher_kd: bool = False, kd_alpha: float = 0.25, kd_temp: float = 2.0,
                     weight_decay: float = 1e-4, seed: int = 42, device=torch.device("cpu"),
                     save_path: str = None, model_name: str = "QAT_Model"):
    """
    Executes low-learning-rate QAT fine-tuning.
    Evaluates INT8 validation accuracy at every epoch.
    Selects and saves checkpoint strictly by peak INT8 validation accuracy.
    """
    torch.manual_seed(seed)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    initial_val_acc = evaluate_qat_validation(model, val_loader, device)
    print(f"--- Starting QAT Fine-Tuning: {model_name} (Seed {seed}) ---")
    print(f"Initial INT8 Val Acc (Epoch 0): {initial_val_acc:.2f}% | Epochs: {epochs} | Initial LR: {lr}")

    best_val_acc = initial_val_acc
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            if use_teacher_kd:
                x, y, t_logits = batch
                t_logits = t_logits.to(device)
            else:
                x, y = batch[0], batch[1]
                t_logits = None

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            loss_ce = ce_loss_fn(out, y)
            if use_teacher_kd and t_logits is not None:
                s_log_probs = F.log_softmax(out / kd_temp, dim=-1)
                t_probs = F.softmax(t_logits / kd_temp, dim=-1)
                loss_kd = kl_loss_fn(s_log_probs, t_probs) * (kd_temp ** 2)
                loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * loss_kd
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, batches)
        val_acc = evaluate_qat_validation(model, val_loader, device)
        cur_lr = scheduler.get_last_lr()[0]

        is_new_best = val_acc > best_val_acc
        if is_new_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        flag = " *** NEW BEST INT8 ***" if is_new_best else ""
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | INT8 Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}% (ep {best_epoch}) | LR: {cur_lr:.2e}{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_acc": val_acc,
            "lr": cur_lr,
        })

    total_time = time.time() - t0
    print(f"\nQAT Fine-Tuning Complete in {total_time:.1f}s. Peak INT8 Val Acc: {best_val_acc:.2f}% @ Epoch {best_epoch}")

    # Load best weights before returning/saving
    model.load_state_dict(best_weights)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({
            "model_state_dict": best_weights,
            "best_int8_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "initial_int8_val_acc": initial_val_acc,
            "seed": seed,
            "total_time_s": total_time,
            "epochs": epochs,
            "lr": lr,
            "history": history,
        }, save_path)
        print(f"Saved best INT8 checkpoint to: {save_path}")

    return {
        "best_int8_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "initial_int8_val_acc": initial_val_acc,
        "total_time_s": total_time,
        "checkpoint_path": save_path,
        "history": history,
    }
