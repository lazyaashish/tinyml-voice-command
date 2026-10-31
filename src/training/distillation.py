"""
Knowledge Distillation Module for Keyword Spotting (KWS)
Allows a pre-trained Teacher model (e.g. Arm Small DS-CNN on 10 MFCCs)
to supervise a lightweight Student model (e.g. Progressive DS-CNN on 40 Log-Mel CMN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DistillationLoss(nn.Module):
    """
    Distillation loss combining Task Cross-Entropy with KL Divergence between
    softened teacher and student output distributions.
    
    Formula:
      L = (1 - alpha) * CE(student, y_true) + alpha * (T^2) * KL(log_softmax(student/T), softmax(teacher/T))
    """
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard task loss with optional label smoothing
        ce_loss = F.cross_entropy(student_logits, targets, label_smoothing=self.label_smoothing)
        
        # Soft distillation loss
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        return total_loss, ce_loss, kl_loss
