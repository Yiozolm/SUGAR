import torch
import torch.nn as nn
from enum import Enum
from typing import Any, Optional, Tuple
from tricode import _NeLU_triton_backward, _BSiLU_triton_backward, _relu_triton_forward


class SurrogateType(Enum):
    BSILU = "BSiLU"
    NELU = "NeLU"

class BSiLU(nn.Module):
    def __init__(self, alpha: float = 1.67) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) + (x + self.alpha) * torch.sigmoid(x) * (1 - torch.sigmoid(x))

class NeLU(nn.Module):
    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, torch.tensor(1.0, device=x.device), self.alpha * (2 * x) / (1 + x**2)**2)

class SUGARFunction(torch.autograd.Function):
    """
    Applies the SUGAR method to ReLU activation.
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, surrogate_type: SurrogateType, use_triton=False) -> torch.Tensor:
        """
        Forward pass: Applies ReLU.
        """
        ctx.surrogate_type = surrogate_type
        ctx.use_triton = use_triton
        ctx.save_for_backward(x)
        if use_triton:
            return _relu_triton_forward(x)
        else:
            return torch.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """
        Backward pass: Overrides ReLU's gradient with the surrogate gradient.
        """
        x, = ctx.saved_tensors
        if ctx.surrogate_type == SurrogateType.BSILU:
            if ctx.use_triton:
                grad_x = _BSiLU_triton_backward(x) * grad_output
            else:
                grad_x = BSiLU().gradient(x) * grad_output
        elif ctx.surrogate_type == SurrogateType.NELU:
            if ctx.use_triton:
                grad_x = _NeLU_triton_backward(x) * grad_output
            else:
                grad_x = NeLU().gradient(x) * grad_output
        else:
            raise ValueError(f"Unsupported surrogate type: {ctx.surrogate_type}")
        return grad_x, None, None  # None for the surrogate_type argument

class SUGAR(nn.Module):
    def __init__(self, surrogate_type: SurrogateType) -> None:
        super().__init__()
        self.surrogate_type = surrogate_type

    def forward(self, x: torch.Tensor, use_triton=False) -> torch.Tensor:
        return SUGARFunction.apply(x, self.surrogate_type, use_triton)

if __name__ == '__main__':
    x = torch.randn(10, requires_grad=True)

    # Example usage with BSiLU
    sugar_bsilu = SUGAR(SurrogateType.BSILU)
    output_bsilu = sugar_bsilu(x)
    output_bsilu.sum().backward()
    print("BSiLU torch gradients:", x.grad)
    output_bsilu_triton = sugar_bsilu(x, use_triton=True)
    output_bsilu_triton.sum().backward()
    print("BSiLU triton gradients:", x.grad)

    # Example usage with NeLU
    # x.grad.zero_()  # Reset gradients
    # sugar_nelu = SUGAR(SurrogateType.NELU)
    # output_nelu = sugar_nelu(x)
    # output_nelu.sum().backward()
    # print("NeLU gradients:", x.grad)