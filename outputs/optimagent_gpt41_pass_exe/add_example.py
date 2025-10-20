import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'VECTOR_SIZE': 1}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64, 'VECTOR_SIZE': 2}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 4}, num_warps=8, num_stages=5),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
    VECTOR_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load for better memory throughput
    if VECTOR_SIZE > 1:
        x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        y = tl.load(in_ptr1 + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    else:
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
    
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add_wrapper(x, y):
    out = torch.zeros_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n_elements)
    return out