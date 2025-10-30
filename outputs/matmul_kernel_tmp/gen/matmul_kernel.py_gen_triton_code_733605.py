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
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


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

##################################################################################################################################################





def test_matmul():

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 64

    M, N, K = 4096, 4096, 4096



    # Initialize matrices on CUDA device

    c = torch.empty((M, N), device='cuda:0', dtype=torch.float16)

    a = torch.rand((M, K), device='cuda:0', dtype=torch.float16)

    b = torch.rand((K, N), device='cuda:0', dtype=torch.float16)



    # Call the matmul function multiple times

    test_case_1 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)



    # Additional test cases to cover more branches

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 64, 128

    test_case_2 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)



    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 256, 64

    test_case_3 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)



    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32

    test_case_4 = matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)



    return {

        "test_case_1": test_case_1,

        "test_case_2": test_case_2,

        "test_case_3": test_case_3,

        "test_case_4": test_case_4

    }



    # # Call the matmul function multiple times

    # matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # test_case_1 = c.clone()



    # # Additional test cases to cover more branches

    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 64, 128

    # matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # test_case_2 = c.clone()



    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 256, 64

    # matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # test_case_3 = c.clone()



    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32

    # matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # test_case_4 = c.clone()



    # return {

    #     "test_case_1": test_case_1,

    #     "test_case_2": test_case_2,

    #     "test_case_3": test_case_3,

    #     "test_case_4": test_case_4

    # }



result_gold = test_matmul()
