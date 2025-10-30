"""
RAG Configuration for Triton Kernel Generation
"""

# RAG Context Length Limits (characters)
RAG_CONTEXT_LIMITS = {
    'hardware_context': 1200,
    'tutorial_context': 1200, 
    'optimization_context': 1200,
    'max_total_context': 4000  # Total context limit
}

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Fast and efficient
    'device': 'cpu',
    'cache_folder': '.rag_cache'
}

# Document Processing Configuration
DOCUMENT_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'top_k_per_source': 3,
    'similarity_threshold': 0.1  # Minimum similarity for inclusion
}

# Operator Classification for Better Retrieval
OPERATOR_MAPPINGS = {
    # Matrix operations
    'matmul': ['matrix multiplication', 'gemm', 'dot product', 'tensor core'],
    'bmm': ['batch matrix multiplication', 'batched gemm'],
    
    # Normalization
    'layernorm': ['layer normalization', 'mean variance', 'affine transform'],
    'rmsnorm': ['rms normalization', 'root mean square'],
    'batchnorm': ['batch normalization'],
    
    # Activation functions
    'softmax': ['softmax', 'attention weights', 'probability distribution'],
    'relu': ['rectified linear', 'activation'],
    'gelu': ['gaussian error linear', 'activation'],
    'silu': ['swish', 'sigmoid linear'],
    
    # Attention mechanisms
    'attention': ['scaled dot product', 'flash attention', 'self attention'],
    'flash_attn': ['flash attention', 'memory efficient attention'],
    
    # Convolution
    'conv': ['convolution', 'filter', 'kernel', 'im2col'],
    'depthwise': ['depthwise convolution', 'channel-wise'],
    
    # Reduction operations
    'reduce': ['reduction', 'sum', 'max', 'min', 'mean'],
    'cumsum': ['cumulative sum', 'prefix sum'],
    
    # Memory operations
    'transpose': ['transpose', 'permute', 'swizzle'],
    'copy': ['memory copy', 'data movement'],
    'gather': ['gather', 'index select'],
    'scatter': ['scatter', 'index write'],
    
    # Embedding and lookup
    'embedding': ['embedding lookup', 'table lookup', 'gather'],
    
    # Quantization
    'quantize': ['quantization', 'int8', 'fp16', 'precision'],
    'dequantize': ['dequantization', 'upcast'],
}

# Hardware-specific query hints
HARDWARE_QUERY_HINTS = {
    'matmul': ['tensor core', 'wmma', 'mma', 'shared memory', 'occupancy'],
    'elementwise': ['coalescing', 'vectorization', 'memory bandwidth'],
    'layernorm': ['shared memory', 'warp reduce', 'numerical stability'],
    'softmax': ['rowwise reduce', 'max-sub trick', 'overflow prevention'],
    'attention': ['qkv layout', 'tiling', 'memory reuse', 'pipeline'],
    'conv': ['shared memory reuse', 'bank conflict', 'tile swizzle'],
    'reduce': ['warp shuffle', 'tree reduce', 'atomic operations'],
    'transpose': ['swizzle pattern', 'coalescing', 'bank conflict'],
}

# Optimization-specific query hints  
OPTIMIZATION_QUERY_HINTS = {
    'matmul': ['block size', 'tiling', 'num warps', 'tensor core usage'],
    'elementwise': ['vector width', 'coalesced access', 'masking'],
    'layernorm': ['fused epilogue', 'shared memory reduction'],
    'softmax': ['online reduce', 'numerical stability', 'tile size'],
    'attention': ['io aware', 'blocking strategy', 'pipeline'],
    'conv': ['tiling strategy', 'preload', 'memory reuse'],
    'reduce': ['warp shuffle', 'partial sums', 'block reduce'],
    'transpose': ['swizzle', 'vectorized load/store'],
}

def get_operator_class(filename: str) -> str:
    """Extract operator class from filename"""
    name = filename.lower().replace('.py', '')
    
    # Remove common suffixes
    for suffix in ['_triton', '_kernel', '_fwd', '_bwd', '_forward', '_backward']:
        name = name.replace(suffix, '')
    
    # Check against known mappings
    for op_class, keywords in OPERATOR_MAPPINGS.items():
        if any(keyword.replace(' ', '_') in name or keyword.replace(' ', '') in name 
               for keyword in [op_class] + keywords):
            return op_class
    
    # Fallback to first part of name
    return name.split('_')[0] if '_' in name else name

def build_enhanced_query(op_class: str, instruction: str, query_type: str) -> str:
    """Build enhanced query for specific document type"""
    base_terms = [op_class]
    
    # Add operator-specific terms
    if op_class in OPERATOR_MAPPINGS:
        base_terms.extend(OPERATOR_MAPPINGS[op_class][:3])  # Top 3 terms
    
    # Add type-specific hints
    if query_type == 'hardware' and op_class in HARDWARE_QUERY_HINTS:
        base_terms.extend(HARDWARE_QUERY_HINTS[op_class][:3])
    elif query_type == 'optimization' and op_class in OPTIMIZATION_QUERY_HINTS:
        base_terms.extend(OPTIMIZATION_QUERY_HINTS[op_class][:3])
    
    # Add instruction context
    if instruction:
        base_terms.append(instruction[:100])  # First 100 chars
    
    return ' '.join(base_terms)