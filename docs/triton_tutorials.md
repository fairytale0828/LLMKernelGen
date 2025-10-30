# Triton Tutorial: Standard Operator Implementations

## Basic Vector Operations

### Element-wise Addition
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Softmax Implementation
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Write output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

## Matrix Operations

### Matrix Multiplication (GEMM)
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### Layer Normalization
```python
@triton.jit
def layernorm_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    stride, N, eps, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    output_ptr += row * stride
    input_ptr += row * stride
    
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=0., eviction_policy="evict_last")
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=0., eviction_policy="evict_last")
        a = tl.where(cols < N, a - mean, 0.)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Store mean and rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    
    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(weight_ptr + cols, mask=mask)
        bias = tl.load(bias_ptr + cols, mask=mask)
        a = tl.load(input_ptr + cols, mask=mask, other=0., eviction_policy="evict_first")
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        tl.store(output_ptr + cols, out, mask=mask)
```

## Attention Mechanisms

### Flash Attention (Simplified)
```python
@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    L, M,  # L: logsumexp, M: max
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, P_SEQ, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize pointers
    q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs)
    
    # Loop over K, V
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= 1.44269504  # 1/ln(2)
        
        # Apply causal mask
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        # Update statistics
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update accumulator
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p, v)
        
        l_i = l_i_new
        m_i = m_i_new
    
    # Store output
    offs_n = tl.arange(0, HEAD_DIM)
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc)
```

## Reduction Operations

### Sum Reduction
```python
@triton.jit
def sum_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * N
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Reduce along row
    for col_idx in range(0, N, BLOCK_SIZE):
        cols = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        acc += vals
    
    # Final reduction
    result = tl.sum(acc)
    tl.store(output_ptr + pid, result)
```

## Activation Functions

### ReLU
```python
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
```

### GELU
```python
@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/π) ≈ 0.7978845608
    tanh_inner = tl.libdevice.tanh(inner)
    output = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Memory Access Patterns

### Transpose
```python
@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load input
    x = tl.load(input_ptrs, mask=mask)
    
    # Store transposed
    output_ptrs = output_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    tl.store(output_ptrs, x.T, mask=mask.T)
```

## Best Practices

### Memory Coalescing
```python
# Good: Coalesced access
offsets = tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)

# Bad: Strided access
offsets = tl.arange(0, BLOCK_SIZE) * stride
data = tl.load(ptr + offsets)
```

### Shared Memory Usage
```python
# Allocate shared memory for data reuse
shared_data = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
```

### Vectorization
```python
# Use vector operations instead of loops
result = tl.sum(data, axis=0)  # Good
# vs manual loop (Bad)
```