# RAG Integration for Triton Kernel Generation

## Overview

This enhancement adds Agentic RAG (Retrieval-Augmented Generation) capabilities to the LLMKernelGen project using LangChain. The system now retrieves relevant context from three knowledge sources during code generation:

1. **Hardware Characteristics** - GPU architecture, memory hierarchy, performance considerations
2. **Triton Tutorials** - Standard operator implementations and best practices  
3. **Optimization Techniques** - Performance optimization strategies and patterns

## Installation

### Install Additional Dependencies

```bash
# Install RAG dependencies
pip install -r requirements_rag.txt

# Or install individually
pip install langchain==0.1.0 faiss-cpu==1.7.4 sentence-transformers==2.2.2
```

### Verify Installation

```bash
cd LLMKernelGen
python test_rag_integration.py
```

## Usage

### Basic Usage

The RAG functionality is automatically integrated into the existing OptimAgent. No code changes required for basic usage:

```bash
cd src
python main_optimagent_tritonbench.py
```

### Configuration

Update your config file to customize RAG behavior:

```yaml
# RAG settings
docs_path: "../docs"  # Path to knowledge base documents
enable_rag: true      # Enable/disable RAG functionality
```

### Custom Documents

Add your own knowledge documents to the `docs/` directory:

```
docs/
├── hardware_characteristics.md    # GPU hardware info
├── triton_tutorials.md           # Standard implementations
├── optimization_techniques.md     # Optimization strategies
└── custom_knowledge.md           # Your custom content
```

## How It Works

### RAG Integration Flow

1. **Initialization**: RAG retriever loads and indexes documents using sentence transformers
2. **Caching**: Vector databases are cached locally in `.rag_cache/` for faster subsequent runs
3. **Context Retrieval**: For each operator, relevant context is retrieved based on:
   - Operator name (e.g., "matmul", "layernorm")
   - Task instruction content
   - Error messages (for debugging)
4. **Prompt Enhancement**: Retrieved context is added to generation prompts
5. **Usage Tracking**: Detailed logs and reports track RAG usage
6. **Iterative Improvement**: Context helps guide optimization across iterations

### Storage and Caching

**Offline RAG Features:**
- Vector databases cached in `.rag_cache/` directory
- Automatic cache invalidation when source documents change
- ~10x faster startup after initial indexing
- Embeddings downloaded once and cached by HuggingFace

**Storage Locations:**
```
LLMKernelGen/
├── .rag_cache/                    # Vector database cache
│   ├── hardware_vectorstore/      # Hardware docs FAISS index
│   ├── tutorials_vectorstore/     # Tutorial docs FAISS index
│   └── optimization_vectorstore/  # Optimization docs FAISS index
├── outputs/
│   ├── experiment_rag_usage.json  # RAG usage report
│   └── experiment.jsonl           # Main results
└── ~/.cache/huggingface/          # Embedding models cache
```

### Example Context Retrieval

For `matmul_triton.py`:
- **Hardware Context**: Memory bandwidth, tensor cores, occupancy guidelines
- **Tutorial Context**: Standard GEMM implementation patterns
- **Optimization Context**: Tiling strategies, memory coalescing techniques

## Architecture

### Components

```
LLMKernelGen/
├── src/
│   ├── retrievers/
│   │   └── langchain_rag.py      # RAG retriever implementation
│   └── agents/
│       └── OptimAgent.py         # Enhanced with RAG integration
├── docs/                         # Knowledge base documents
│   ├── hardware_characteristics.md
│   ├── triton_tutorials.md
│   └── optimization_techniques.md
└── test_rag_integration.py       # Test script
```

### Key Classes

- `TritonRAGRetriever`: LangChain-based retriever with FAISS vectorstore
- `FallbackRAGRetriever`: Simple keyword-based fallback when LangChain unavailable
- `OptimAgent`: Enhanced with `_get_rag_context()` method

## Performance Impact

### Benefits
- **Improved First-Attempt Success**: Better initial code generation with relevant examples
- **Faster Convergence**: Optimization guidance reduces iteration count
- **Better Error Recovery**: Hardware-aware debugging suggestions

### Overhead
- **Initialization**: ~2-3 seconds for document indexing
- **Per-Query**: ~50-100ms for context retrieval
- **Memory**: ~100MB additional for embeddings and vectorstore

## Troubleshooting

### Common Issues

1. **LangChain Import Error**
   ```bash
   pip install langchain faiss-cpu sentence-transformers
   ```

2. **Embedding Model Download**
   ```python
   # First run downloads ~90MB model
   # Subsequent runs use cached model
   ```

3. **Document Not Found**
   ```
   Warning: hardware_characteristics.md not found in docs/
   ```
   Ensure documents exist in the specified `docs_path`

### Fallback Mode

If LangChain dependencies are unavailable, the system automatically falls back to:
- Simple keyword-based document search
- Reduced functionality but no crashes
- Warning messages in logs

## Customization

### Adding New Document Types

1. Create new markdown file in `docs/`
2. Update `TritonRAGRetriever._load_documents()` to include new type
3. Add query method for new document type

### Custom Embedding Models

```python
# In langchain_rag.py
retriever = TritonRAGRetriever(
    docs_path="docs",
    embedding_model="all-mpnet-base-v2"  # Different model
)
```

### Adjusting Context Length

```python
# In OptimAgent._get_rag_context()
context['hardware_context'][:1200]  # Increase from 800
```

## Examples

### Generated Context Example

For `layernorm_fwd_triton.py`:

```
## Additional Context from Knowledge Base:

### Hardware Characteristics:
Shared Memory: 48KB-164KB per SM, ~19TB/s bandwidth, 1-2 cycle latency
Use for data reuse patterns in normalization operations...

### Reference Implementation:
@triton.jit
def layernorm_kernel(output_ptr, input_ptr, weight_ptr, bias_ptr...
Compute mean and variance in separate passes for numerical stability...

### Optimization Techniques:
Memory Coalescing: Ensure adjacent threads access adjacent elements
Block Size: Choose 128-256 for normalization kernels...
```

## Future Enhancements

- [ ] Dynamic document updates during runtime
- [ ] Operator-specific fine-tuned embeddings
- [ ] Integration with code execution feedback
- [ ] Multi-modal retrieval (code + documentation)
- [ ] Automatic knowledge base expansion from successful optimizations
## M
onitoring RAG Usage

### Real-time Monitoring

During execution, you'll see logs like:
```
INFO - [matmul_triton.py] 🔍 RAG Enhanced: Hardware: 1200 chars, Tutorial: 800 chars, Optimization: 600 chars
INFO - [layernorm_fwd_triton.py] ⚠️  No relevant RAG context found
INFO - [softmax_triton.py] 🔍 RAG Enhanced: Tutorial: 900 chars, Optimization: 750 chars
```

### Post-Run Analysis

After completion, check:

1. **Console Summary**:
```
📊 RAG USAGE SUMMARY
✅ matmul_triton.py: 3 iterations with RAG
✅ layernorm_fwd_triton.py: 2 iterations with RAG
❌ custom_kernel.py: No RAG usage
📈 Overall: 2/3 operators used RAG
📚 Source Usage: Hardware=5, Tutorial=4, Optimization=6
```

2. **Detailed Report** (`outputs/experiment_rag_usage.json`):
```json
{
  "summary": {
    "total_operators": 3,
    "rag_enabled_operators": 2,
    "total_rag_retrievals": 5,
    "avg_context_length": 850
  },
  "per_operator": {
    "matmul_triton.py": {
      "iterations_with_rag": 3,
      "total_context_chars": 2400,
      "sources_used": {"Hardware": 2, "Tutorial": 1, "Optimization": 3},
      "usage_per_iteration": [...]
    }
  }
}
```

3. **Analysis Tool**:
```bash
python analyze_rag_usage.py outputs/experiment_rag_usage.json
```

### Verification Methods

**Method 1: Check Logs**
```bash
# Look for RAG enhancement messages
grep "RAG Enhanced" logs/experiment.log

# Count RAG usage
grep -c "🔍 RAG Enhanced" logs/experiment.log
```

**Method 2: Inspect Generated Prompts**
Add debug logging to see actual prompts sent to LLM:
```python
# In OptimAgent.generate_solution(), add:
logger.debug(f"Full prompt for {mem.ps.filename}:\n{text}")
```

**Method 3: Compare Performance**
```bash
# Run without RAG
python main_optimagent_tritonbench.py --disable-rag

# Run with RAG  
python main_optimagent_tritonbench.py

# Compare results
python analyze_rag_usage.py outputs/with_rag_usage.json
```

## Troubleshooting RAG Issues

### No RAG Context Found
**Symptoms**: Logs show "⚠️ No relevant RAG context found"
**Causes**:
- Document doesn't contain relevant keywords
- Operator name not in mapping dictionary
- Embedding similarity too low

**Solutions**:
```python
# Add operator mapping in langchain_rag.py
operator_mapping = {
    'your_custom_op': 'custom operation description',
    # ...
}

# Or add relevant keywords to documents
```

### Cache Issues
**Symptoms**: "Failed to load cache" warnings
**Solutions**:
```bash
# Clear cache and rebuild
rm -rf .rag_cache/
python main_optimagent_tritonbench.py
```

### Memory Usage
**Symptoms**: High memory usage during startup
**Solutions**:
- Use smaller embedding model: `all-MiniLM-L6-v2` (default) vs `all-mpnet-base-v2`
- Reduce chunk size in text splitter
- Disable caching: `use_cache=False`