import triton
import torch
from func import SUGAR, SurrogateType

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  
        line_arg='provider',  
        line_vals=['triton', 'torch'], 
        line_names=[
            "Triton",
            "Torch",
        ],  
        styles=[('blue', '-'), ('green', '-')],  
        ylabel="GB/s", 
        plot_name="SUGAR-performance", 
        args={'M': 4096}, 
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: SUGAR(SurrogateType.BSILU)(x))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: SUGAR(SurrogateType.BSILU)(x, use_triton=True))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)