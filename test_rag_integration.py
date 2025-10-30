#!/usr/bin/env python3
"""
Test script for RAG integration
"""

import sys
import os
sys.path.append('src')
from retrievers.langchain_rag import create_rag_retriever

def test_rag_retriever():
    """Test RAG retriever functionality"""
    print("=== Testing RAG Retriever ===")
    
    # Create retriever
    try:
        retriever = create_rag_retriever(docs_path="docs")  # Fixed path
        print("✓ RAG retriever created successfully")
    except Exception as e:
        print(f"✗ Failed to create RAG retriever: {e}")
        return False
    
    # Test operator context retrieval
    test_cases = [
        ("matmul_triton.py", "Implement matrix multiplication using Triton"),
        ("layernorm_fwd_triton.py", "Implement layer normalization forward pass"),
        ("softmax_triton.py", "Implement softmax activation function"),
    ]
    
    for operator_name, instruction in test_cases:
        print(f"\n--- Testing {operator_name} ---")
        try:
            context = retriever.get_context_for_operator(operator_name, instruction)
            
            for context_type, content in context.items():
                if content:
                    print(f"✓ {context_type}: {len(content)} characters")
                    # Show first 100 characters
                    preview = content[:100].replace('\n', ' ')
                    print(f"  Preview: {preview}...")
                else:
                    print(f"- {context_type}: No content")
                    
        except Exception as e:
            print(f"✗ Error getting context for {operator_name}: {e}")
    
    return True

def test_fallback_retriever():
    """Test fallback retriever when LangChain is not available"""
    print("\n=== Testing Fallback Retriever ===")
    
    from retrievers.langchain_rag import FallbackRAGRetriever
    
    try:
        retriever = FallbackRAGRetriever(docs_path="docs")
        print("✓ Fallback retriever created successfully")
        
        context = retriever.get_context_for_operator("matmul.py", "matrix multiplication")
        
        for context_type, content in context.items():
            if content:
                print(f"✓ {context_type}: {len(content)} characters")
            else:
                print(f"- {context_type}: No content")
                
        return True
        
    except Exception as e:
        print(f"✗ Fallback retriever failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Integration for Triton Kernel Generation")
    print("=" * 50)
    
    # Test main retriever
    success1 = test_rag_retriever()
    
    # Test fallback
    success2 = test_fallback_retriever()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)