import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    c_ptr,
    a_ptr,
    b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate starting offsets for the output block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Precompute number of K blocks
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Initialize pointers for A and B
    a_block_ptr = a_ptr + offs_m[:, None] * stride_am
    b_block_ptr = b_ptr + offs_n[None, :] * stride_bn
    
    # Main computation loop
    for k_block in range(0, num_k_blocks):
        k_offset = k_block * BLOCK_SIZE_K
        
        # Load A block with masking
        a_ptrs = a_block_ptr + (k_offset + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & ((k_offset + tl.arange(0, BLOCK_SIZE_K))[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with masking
        b_ptrs = b_block_ptr + (k_offset + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk
        b_mask = ((k_offset + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) & (offs_n[None, :] < N)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix product
        accumulator += tl.dot(a_block, b_block, allow_tf32=True)
    
    # Convert and store result
    c_block = accumulator.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_block, mask=c_mask)


def matmul(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int):
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        c_ptr=c,
        a_ptr=a,
        b_ptr=b,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c