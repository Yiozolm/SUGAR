import triton
import triton.language as tl
from triton.runtime import driver
import torch

@triton.jit
def _relu_kernel_forward(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(input_ptr + block_start, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(output_ptr + block_start, y, mask=mask)

def _relu_triton_forward(input: torch.Tensor)-> torch.Tensor:
    output: torch.Tensor = torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    n_elements: int = input.numel()
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _relu_kernel_forward[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def _BSiLU_kernel_backward(input_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(input_ptr + block_start, mask=mask)
    soft = tl.sigmoid(x)
    y = soft + (x + alpha) * soft * (1 - soft)
    tl.store(output_ptr + block_start, y, mask=mask)

def _BSiLU_triton_backward(input:torch.Tensor, alpha:float=1.67)-> torch.Tensor:
    output: torch.Tensor = torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    n_elements: int = input.numel()
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _BSiLU_kernel_backward[grid](input, output, alpha, n_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def _NeLU_kernel_backward(input_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(input_ptr + block_start, mask=mask)
    y = tl.where(x > 0, 1.0, alpha * 2 * x / x**2)
    tl.store(output_ptr + block_start, y, mask=mask)

def _NeLU_triton_backward(input:torch.Tensor, alpha:float=0.2)-> torch.Tensor:
    output: torch.Tensor = torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    n_elements: int = input.numel()
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _NeLU_kernel_backward[grid](input, output, alpha, n_elements, BLOCK_SIZE=1024)
    return output