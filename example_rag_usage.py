#!/usr/bin/env python3
"""
Example usage of RAG-enhanced Triton kernel generation
"""

import sys
import os
sys.path.append('src')

from retrievers.langchain_rag import create_rag_retriever

def demonstrate_rag_retrieval():
    """Demonstrate RAG retrieval for different operator types"""
    
    print("🔍 RAG-Enhanced Triton Kernel Generation Demo")
    print("=" * 50)
    
    # Initialize RAG retriever
    print("Initializing RAG retriever...")
    retriever = create_rag_retriever(docs_path="docs")
    print("✓ RAG retriever initialized\n")
    
    # Test different operator types
    test_operators = [
        {
            'name': 'matmul_triton.py',
            'instruction': 'Implement efficient matrix multiplication using Triton with tensor core optimization',
            'description': 'Matrix Multiplication'
        },
        {
            'name': 'layernorm_fwd_triton.py', 
            'instruction': 'Implement layer normalization forward pass with fused mean and variance computation',
            'description': 'Layer Normalization'
        },
        {
            'name': 'flash_attention.py',
            'instruction': 'Implement memory-efficient attention mechanism using block-wise computation',
            'description': 'Flash Attention'
        },
        {
            'name': 'softmax_triton.py',
            'instruction': 'Implement numerically stable softmax with online reduction',
            'description': 'Softmax Activation'
        }
    ]
    
    for i, op in enumerate(test_operators, 1):
        print(f"{i}. Testing {op['description']} ({op['name']})")
        print("-" * 40)
        
        # Get RAG context
        context = retriever.get_context_for_operator(op['name'], op['instruction'])
        
        # Display context summary
        total_chars = 0
        for context_type, content in context.items():
            char_count = len(content)
            total_chars += char_count
            
            if char_count > 0:
                print(f"  ✓ {context_type.replace('_', ' ').title()}: {char_count:,} characters")
                
                # Show a preview of the content
                preview = content[:150].replace('\n', ' ').strip()
                if len(content) > 150:
                    preview += "..."
                print(f"    Preview: {preview}")
            else:
                print(f"  - {context_type.replace('_', ' ').title()}: No content found")
        
        print(f"  📊 Total Context: {total_chars:,} characters")
        
        # Simulate how this would be used in prompt
        if total_chars > 0:
            print(f"  💡 This context would be added to the LLM prompt for better code generation")
        else:
            print(f"  ⚠️  No RAG context available - falling back to base generation")
        
        print()
    
    print("Demo completed! 🎉")
    print("\nNext steps:")
    print("1. Run OptimAgent with RAG enabled to see real improvements")
    print("2. Check generated RAG usage reports for detailed analytics")
    print("3. Use rag_diagnostics.py to monitor and optimize RAG performance")

def show_rag_integration_in_prompt():
    """Show how RAG context integrates into the actual prompt"""
    
    print("\n" + "=" * 60)
    print("📝 RAG INTEGRATION IN PROMPTS")
    print("=" * 60)
    
    # Get sample context
    retriever = create_rag_retriever(docs_path="docs")
    context = retriever.get_context_for_operator(
        "matmul_triton.py", 
        "Implement matrix multiplication kernel"
    )
    
    # Show base prompt structure
    base_prompt = """
Implement a Triton kernel for matrix multiplication.

Requirements:
- Use efficient memory access patterns
- Optimize for GPU hardware
- Handle different matrix sizes
"""
    
    # Show enhanced prompt with RAG
    enhanced_prompt = base_prompt + "\n\n## Additional Context from Knowledge Base:\n"
    
    if context.get('hardware_context'):
        enhanced_prompt += f"### Hardware Characteristics:\n{context['hardware_context'][:300]}...\n\n"
    
    if context.get('tutorial_context'):
        enhanced_prompt += f"### Reference Implementation:\n{context['tutorial_context'][:300]}...\n\n"
    
    if context.get('optimization_context'):
        enhanced_prompt += f"### Optimization Techniques:\n{context['optimization_context'][:300]}...\n\n"
    
    print("🔸 Base Prompt (without RAG):")
    print(base_prompt)
    print(f"   Length: {len(base_prompt)} characters")
    
    print("\n🔹 Enhanced Prompt (with RAG):")
    print(enhanced_prompt[:800] + "..." if len(enhanced_prompt) > 800 else enhanced_prompt)
    print(f"   Length: {len(enhanced_prompt)} characters")
    print(f"   Enhancement: +{len(enhanced_prompt) - len(base_prompt)} characters of context")
    
    print("\n💡 Benefits of RAG Enhancement:")
    print("   • Hardware-aware optimizations")
    print("   • Reference implementations for guidance")
    print("   • Proven optimization techniques")
    print("   • Better first-attempt success rate")
    print("   • Faster convergence in iterative optimization")

if __name__ == "__main__":
    try:
        demonstrate_rag_retrieval()
        show_rag_integration_in_prompt()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that docs/ directory exists with required markdown files")
        print("3. Run: python rag_diagnostics.py --setup-only")
        sys.exit(1)