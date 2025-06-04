import triton
import triton.language as tl
from triton.runtime import driver
import torch


device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8
    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    # Allocate output
    y = torch.empty_like(x)
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y



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
def _BSiLU_kernel_backward(input_ptr, output_ptr, alpha, soft, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(input_ptr + block_start, mask=mask)
    y = soft + (x + alpha) * soft * (1-soft)
    tl.store(output_ptr + block_start, y, mask=mask)

def _BSiLU_triton_backward(input:torch.Tensor, alpha:float=1.67)-> torch.Tensor:
    output: torch.Tensor = torch.empty_like(input)
    if not input.is_contiguous():
        input = input.contiguous()
    n_elements: int = input.numel()
    soft = softmax(input)
    grid = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _BSiLU_kernel_backward[grid](input, output, alpha, soft, n_elements, BLOCK_SIZE=1024)
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