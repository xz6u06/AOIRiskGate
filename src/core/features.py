from __future__ import annotations
from typing import Dict, List, Tuple
import torch


def attach_activation_hook(model, module, name: str = "feat"):
    activation: Dict[str, List[torch.Tensor]] = {name: []}

    def hook(_m, _inp, out):
        activation[name].append(out.detach().cpu())

    handle = module.register_forward_hook(hook)
    return activation, handle


def activation_to_embedding(act: torch.Tensor) -> torch.Tensor:
    # avgpool: [N, 512, 1, 1] -> [N, 512]
    return act.flatten(1)
