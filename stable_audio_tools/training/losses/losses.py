import typing as tp
import torch

from torch.nn import functional as F
from torch import nn
from .utils import mmd

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, decay = 1.0):
        super().__init__()

        self.name = name
        self.decay = float(decay)
        self.master_weight = weight
        weight = torch.tensor(float(weight))
        self.register_buffer('weight', weight)

    def decay_weight(self):
        if self.decay != 1.0:
            self.weight *= self.decay
        elif self.decay == 1.0 and self.weight != self.master_weight:
            self.weight = torch.tensor(self.master_weight, dtype=self.weight.dtype, device=self.weight.device)
    def forward(self, info, *args, **kwargs):
        raise NotImplementedError

class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0, decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key = key

    def forward(self, info):
        self.decay_weight()
        return self.weight * info[self.key]

class TargetValueLoss(LossModule):
    def __init__(self, key: str, target: float, name: str, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
        self.target = target
    
    def forward(self, info):
        self.decay_weight()
        return self.weight * (info[self.key] - self.target).abs()

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        l1_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info:
            l1_loss = l1_loss[info[self.mask_key]]

        l1_loss = l1_loss.mean()
        self.decay_weight()
        return self.weight * l1_loss

class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss',decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()
        self.decay_weight()
        return self.weight * mse_loss

class LossWithTarget(LossModule):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(name, weight, decay)

        self.loss_module = loss_module

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.loss_module(info[self.input_key], info[self.target_key])
        self.decay_weight()
        return self.weight * loss

class AuralossLoss(LossWithTarget):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(loss_module, input_key=input_key, target_key=target_key, name=name, weight=weight, decay=decay)
    def forward(self, info):
        loss = self.loss_module(info[self.target_key], info[self.input_key]) # Enforce wrong order of input and target until we find issue in Auraloss
        self.decay_weight()
        return self.weight * loss

class MultiLoss(nn.Module):
    def __init__(
        self,
        losses: tp.List[LossModule],
        adaptive_loss_keys: tp.Optional[tp.List[str]] = None,
        adaptive_loss_back_keys: tp.Optional[tp.List[str]] = None,
        
        reference_loss_key: tp.Optional[str] = None,
        model: tp.Optional[nn.Module] = None,
        model_param_names: tp.Optional[tp.List[str]] = None,
        model_param_back_names: tp.Optional[tp.List[str]] = None,
    ):
        super().__init__()

        self.losses = nn.ModuleList(losses)
        self.adaptive_loss_keys = adaptive_loss_keys
        self.adaptive_loss_back_keys = adaptive_loss_back_keys
        self.reference_loss_key = reference_loss_key
        self.model = model
        # e.g., ["encoder.proj_out.weight", "encoder.proj_in.bias"]
        self.model_param_names = model_param_names
        self.model_param_back_names = model_param_back_names

    def forward(self, info):
        total_loss = 0

        losses = {}
        adp_weights = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            train_loss = module_loss
            losses[loss_module.name] = (module_loss, loss_module.weight.item(), train_loss)

        if isinstance(self.adaptive_loss_keys, list) and self.adaptive_loss_keys:
            assert self.reference_loss_key is not None and self.reference_loss_key in losses
            assert self.model_param_names is not None and self.model_param_names
            assert self.model is not None
            params = [
                p for name, p in self.model.named_parameters()
                if name in self.model_param_names
            ]

            loss_ref, weight_ref, _ = losses[self.reference_loss_key]
            loss_ref_orig = loss_ref / weight_ref
            grad_ref = torch.autograd.grad(loss_ref_orig, params, retain_graph=True)
            grad_norm_ref = torch.norm(torch.stack([g.norm() for g in grad_ref]))
            for key in self.adaptive_loss_keys:
                if key in losses:
                    loss_val, weight, train_loss = losses[key]
                    loss_orig = loss_val / weight
                    grad = torch.autograd.grad(loss_orig, params, retain_graph=True)
                    grad_norm = torch.norm(torch.stack([g.norm() for g in grad]))
                    train_loss = loss_val * (grad_norm_ref / (grad_norm + 1e-6))
                    adp_weight = grad_norm_ref / (grad_norm + 1e-6)
                    adp_weights[f"adp_weight_{key}"] = adp_weight.detach().item()
                    losses[key] = (loss_val, weight, train_loss)
        
        if isinstance(self.adaptive_loss_back_keys, list) and self.adaptive_loss_back_keys:
            assert self.reference_loss_key is not None and self.reference_loss_key in losses
            assert self.model_param_back_names is not None and self.model_param_back_names
            assert self.model is not None
            params = [
                p for name, p in self.model.named_parameters()
                if name in self.model_param_back_names
            ]
            loss_ref, weight_ref, _ = losses[self.reference_loss_key]
            loss_ref_orig = loss_ref / weight_ref
            grad_ref = torch.autograd.grad(loss_ref_orig, params, retain_graph=True)
            grad_norm_ref = torch.norm(torch.stack([g.norm() for g in grad_ref]))
            for key in self.adaptive_loss_back_keys:
                if key in losses:
                    loss_val, weight, train_loss = losses[key]
                    loss_orig = loss_val / weight
                    grad = torch.autograd.grad(loss_orig, params, retain_graph=True)
                    grad_norm = torch.norm(torch.stack([g.norm() for g in grad]))
                    train_loss = loss_val * (grad_norm_ref / (grad_norm + 1e-6))
                    adp_weight = grad_norm_ref / (grad_norm + 1e-6)
                    adp_weights[f"adp_weight_{key}"] = adp_weight.detach().item()
                    losses[key] = (loss_val, weight, train_loss)
                    

        losses_for_return = {key: loss_value for key, (loss_value, _, _) in losses.items()}
        for _, _, train_loss in losses.values():
            total_loss += train_loss

        return total_loss, losses_for_return, adp_weights

class StereoImageLoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'stereo_image_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        loss = 0.5*(1 - F.cosine_similarity(info[self.key_a], info[self.key_b], dim=1))

        if self.mask_key is not None and self.mask_key in info:
            loss = loss[info[self.mask_key]]

        loss = loss.mean()
        self.decay_weight()
        return self.weight * loss

class TimeDomainMMDLoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0,  name: str = 'time_domain_mmd_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

    def forward(self, info):
        loss = mmd(info[self.key_a], info[self.key_b], bandwidths=[0.0001,0.001,0.01,0.1,1], dim=-1)
        self.decay_weight()
        return self.weight * loss