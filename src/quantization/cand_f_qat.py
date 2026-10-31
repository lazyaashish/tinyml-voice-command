import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.cand_f import CandF_44ChDualResidual, ConvBNReLU6, DepthwiseSeparableBlock, SEBlock
except ImportError:
    from ..models.cand_f import CandF_44ChDualResidual, ConvBNReLU6, DepthwiseSeparableBlock, SEBlock

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
        # Straight-through estimator: pass gradient where within range
        scaled_x = x / scale
        mask = (scaled_x >= qmin) & (scaled_x <= qmax)
        return grad_output * mask.float(), None, None, None

def fake_quantize_weight(w, qmin=-128, qmax=127):
    # Per-channel symmetric quantization
    if w.dim() == 4:
        scale = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
    else:
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return STEQuantize.apply(w, scale, qmin, qmax)

class FakeQuantizeActivation(nn.Module):
    def __init__(self, qmin=0, qmax=127, momentum=0.1):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer('running_max', torch.tensor(6.0))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def forward(self, x):
        if self.training:
            cur_max = x.detach().abs().max()
            if self.initialized == 0:
                self.running_max.copy_(cur_max)
                self.initialized.fill_(1)
            else:
                self.running_max.copy_(self.running_max * (1 - self.momentum) + cur_max * self.momentum)
        scale = torch.clamp(self.running_max / float(self.qmax), min=1e-8)
        return STEQuantize.apply(x, scale, self.qmin, self.qmax)

class QATConvBNReLU6(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
        self.quant_act = FakeQuantizeActivation(qmin=0, qmax=127)

    def forward(self, x):
        q_w = fake_quantize_weight(self.conv.weight)
        x = F.conv2d(x, q_w, None, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        x = self.bn(x)
        x = self.act(x)
        return self.quant_act(x)

class QATSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_ch = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, mid_ch, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.quant_mid = FakeQuantizeActivation(qmin=0, qmax=127)
        self.fc2 = nn.Linear(mid_ch, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        q_w1 = fake_quantize_weight(self.fc1.weight)
        w = F.linear(w, q_w1, None)
        w = self.act(w)
        w = self.quant_mid(w)
        q_w2 = fake_quantize_weight(self.fc2.weight)
        w = F.linear(w, q_w2, None)
        w = self.sigmoid(w).view(b, c, 1, 1)
        return x * w

class QATDepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=1, padding=1,
                 use_residual: bool = False, use_se: bool = False, se_reduction: int = 8):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch and stride == 1)
        self.dw = QATConvBNReLU6(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pw = QATConvBNReLU6(in_ch, out_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.se = QATSEBlock(out_ch, reduction=se_reduction) if use_se else None
        if self.use_residual:
            self.quant_res = FakeQuantizeActivation(qmin=0, qmax=127)

    def forward(self, x):
        out = self.pw(self.dw(x))
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = self.quant_res(out + x)
        return out

class CandF_QAT(nn.Module):
    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.quant_in = FakeQuantizeActivation(qmin=-128, qmax=127)
        self.init_conv = QATConvBNReLU6(1, 32, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1))
        self.block1 = QATDepthwiseSeparableBlock(32, 32, stride=1, padding=1)
        self.block2 = QATDepthwiseSeparableBlock(32, 44, stride=(2, 1), padding=(1, 1), use_se=True, se_reduction=8)
        self.block3a = QATDepthwiseSeparableBlock(44, 44, stride=1, padding=1, use_residual=True, use_se=True, se_reduction=8)
        self.block3b = QATDepthwiseSeparableBlock(44, 44, stride=1, padding=1, use_residual=True, use_se=True, se_reduction=8)
        self.block4 = QATDepthwiseSeparableBlock(44, 64, stride=1, padding=1, use_se=True, se_reduction=8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.quant_pool = FakeQuantizeActivation(qmin=0, qmax=127)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.quant_in(x)
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.quant_pool(torch.flatten(x, 1))
        q_fc_w = fake_quantize_weight(self.fc.weight)
        return F.linear(x, q_fc_w, self.fc.bias)

def load_fp32_weights_into_qat(qat_model: CandF_QAT, fp32_state_dict: dict):
    """
    Transfers pre-trained FP32 weights into QAT model structure.
    """
    qat_dict = qat_model.state_dict()
    for k, v in fp32_state_dict.items():
        # Map key names
        mapped_k = k
        if 'init_conv.0.' in k: mapped_k = k.replace('init_conv.0.', 'init_conv.conv.')
        elif 'init_conv.1.' in k: mapped_k = k.replace('init_conv.1.', 'init_conv.bn.')
        elif '.dw.0.' in k: mapped_k = k.replace('.dw.0.', '.dw.conv.')
        elif '.dw.1.' in k: mapped_k = k.replace('.dw.1.', '.dw.bn.')
        elif '.pw.0.' in k: mapped_k = k.replace('.pw.0.', '.pw.conv.')
        elif '.pw.1.' in k: mapped_k = k.replace('.pw.1.', '.pw.bn.')

        if mapped_k in qat_dict and qat_dict[mapped_k].shape == v.shape:
            qat_dict[mapped_k].copy_(v)
        else:
            # Check for close matches
            found = False
            for qk in qat_dict.keys():
                if qk.endswith(k) or k.endswith(qk):
                    if qat_dict[qk].shape == v.shape:
                        qat_dict[qk].copy_(v)
                        found = True
                        break

    qat_model.load_state_dict(qat_dict)
    print("Transferred pre-trained FP32 weights to QAT model successfully!")


def train_qat(epochs: int = 15, lr: float = 1e-4, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("STARTING QUANTIZATION-AWARE TRAINING (QAT) ON CandF_44ChDualResidual")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print("=" * 85)

    data_dir = "dataset/speech_commands_v0.02"
    noise_files = [os.path.join(data_dir, "_background_noise_", f) for f in os.listdir(os.path.join(data_dir, "_background_noise_")) if f.endswith('.wav')]
    data_index, words_list, _ = prepare_kws_data_indices(data_dir, ARM_TARGET_WORDS, seed=42)

