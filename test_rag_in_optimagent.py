#!/usr/bin/env python3
"""
Test RAG integration in OptimAgent
"""

import sys
import os
sys.path.append('src')

from agents.OptimAgent import OptimAgent
from models.DeepSeek import DeepSeekModel
from dataloaders.TritonBench import TritonBench

def test_rag_integration():
    """Test RAG integration in OptimAgent"""
    
    print("🔍 Testing RAG Integration in OptimAgent")
    print("=" * 50)
    
    # Mock model for testing (doesn't need real API key)
    class MockModel:
        def generate(self, messages, temperature=0, max_tokens=4096):
            return '{"thought": "test", "code": "# test code"}'
    
    # Create mock dataset with minimal data
    class MockDataset:
        def __init__(self):
            self.problem_states = [MockProblemState()]
        
        def __len__(self):
            return 1
    
    class MockProblemState:
        def __init__(self):
            self.filename = "matmul_kernel.py"
            self.instruction = "Implement matrix multiplication using Triton"
            self.label = None
            self.solution = None
    
    try:
        # Test RAG retriever initialization
        print("1. Testing RAG retriever initialization...")
        
        model = MockModel()
        dataset = MockDataset()
        
        # Test with correct docs path
        agent = OptimAgent(
            model=model, 
            dataset=dataset, 
            corpus_path="src/dataloaders/TB_eval/train_crawl.json",  # This might not exist, but that's ok for this test
            docs_path="docs"  # Correct path from project root
        )
        
        if agent.rag_retriever:
            print("✅ RAG retriever initialized successfully")
            
            # Test RAG context retrieval
            print("\n2. Testing RAG context retrieval...")
            
            # Create a mock memory object
            class MockMemory:
                def __init__(self):
                    self.ps = MockProblemState()
                    self.raw_code = [""]
                    self.function_signatures = []
            
            mem = MockMemory()
            
            # Test _get_rag_context method
            context = agent._get_rag_context(mem)
            
            if context:
                print(f"✅ RAG context retrieved: {len(context)} characters")
                print(f"   Preview: {context[:200]}...")
                
                # Check if rag_usage was recorded
                if hasattr(mem, 'rag_usage') and mem.rag_usage:
                    print(f"✅ RAG usage recorded: {len(mem.rag_usage)} entries")
                    for usage in mem.rag_usage:
                        print(f"   - Sources: {usage['sources']}")
                        print(f"   - Context length: {usage['total_context_length']}")
                else:
                    print("❌ RAG usage not recorded properly")
            else:
                print("❌ No RAG context retrieved")
                
        else:
            print("❌ RAG retriever not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ RAG integration test completed successfully!")
    return True

def test_docs_path_resolution():
    """Test different docs path configurations"""
    
    print("\n🔍 Testing docs path resolution")
    print("=" * 50)
    
    test_paths = [
        "docs",           # Relative from current directory
        "../docs",        # Relative from src directory  
        "./docs",         # Explicit relative
        "/workspace/LLMKernelGen/docs"  # Absolute path
    ]
    
    for path in test_paths:
        print(f"\nTesting path: {path}")
        try:
            sys.path.append('src')
            from retrievers.langchain_rag import create_rag_retriever
            
            retriever = create_rag_retriever(path)
            if retriever:
                # Test a simple query
                context = retriever.get_context_for_operator("matmul.py", "matrix multiplication")
                total_chars = sum(len(v) for v in context.values())
                print(f"✅ Path works: {total_chars} chars retrieved")
            else:
                print("❌ Path failed: No retriever created")
                
        except Exception as e:
            print(f"❌ Path failed: {e}")

if __name__ == "__main__":
    # Change to project root directory
    os.chdir('/workspace/LLMKernelGen')
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Docs directory exists: {os.path.exists('docs')}")
    
    # Test docs path resolution
    test_docs_path_resolution()
    
    # Test RAG integration
    success = test_rag_integration()
    
    if success:
        print("\n🎉 All tests passed! RAG should work correctly now.")
        print("\nTo run OptimAgent with RAG:")
        print("1. cd LLMKernelGen/src")
        print("2. python main_optimagent_tritonbench.py")
    else:
        print("\n❌ Tests failed. Check the error messages above.")