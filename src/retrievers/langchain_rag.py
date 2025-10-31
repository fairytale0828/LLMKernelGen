"""
LangChain-based RAG retriever for Triton kernel optimization
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.schema import BaseRetriever
    
    # Try new imports first
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to old imports
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Install with: pip install langchain langchain-community faiss-cpu sentence-transformers")
    LANGCHAIN_AVAILABLE = False


class TritonRAGRetriever:
    """
    LangChain-based RAG retriever for Triton optimization documents
    """
    
    def __init__(self, docs_path: str = "docs", embedding_model: str = "all-MiniLM-L6-v2", 
                 cache_dir: str = "../.rag_cache", use_cache: bool = True):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain dependencies not available")
            
        self.docs_path = Path(docs_path)
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstores = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Load and index documents (with caching)
        self._load_documents()
    
    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            # Try new HuggingFace embeddings first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
                self.embeddings = HFEmbeddings(
                    model_name=f"sentence-transformers/{self.embedding_model}",
                    model_kwargs={'device': 'cpu'}
                )
            except ImportError:
                # Fallback to community version
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=f"sentence-transformers/{self.embedding_model}",
                    model_kwargs={'device': 'cpu'}
                )
            print(f"✓ Initialized embedding model: {self.embedding_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model {self.embedding_model}: {e}")
            print("Falling back to basic retrieval")
            self.embeddings = None
    
    def _load_documents(self):
        """Load and index documents from docs directory (with caching)"""
        if not self.embeddings:
            return
            
        doc_files = {
            'hardware': 'hardware_characteristics.md',
            'tutorials': 'triton_tutorials.md', 
            'optimization': 'optimization_techniques.md'
        }
        
        for doc_type, filename in doc_files.items():
            file_path = self.docs_path / filename
            cache_path = self.cache_dir / f"{doc_type}_vectorstore"
            
            # Check if cached version exists and is newer than source
            use_cached = (self.use_cache and 
                         cache_path.exists() and 
                         file_path.exists() and
                         cache_path.stat().st_mtime > file_path.stat().st_mtime)
            
            if use_cached:
                try:
                    # Load from cache with security parameter
                    vectorstore = FAISS.load_local(
                        str(cache_path), 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    self.vectorstores[doc_type] = vectorstore
                    print(f"✓ Loaded {doc_type} vectorstore from cache")
                    continue
                except Exception as e:
                    print(f"Warning: Failed to load cache for {doc_type}: {e}")
                    # Fall through to rebuild
            
            # Build vectorstore from source
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split document into chunks
                    documents = [Document(
                        page_content=chunk,
                        metadata={'source': doc_type, 'file': filename}
                    ) for chunk in self.text_splitter.split_text(content)]
                    
                    # Create vector store
                    if documents:
                        vectorstore = FAISS.from_documents(documents, self.embeddings)
                        self.vectorstores[doc_type] = vectorstore
                        print(f"✓ Indexed {len(documents)} chunks from {filename}")
                        
                        # Save to cache
                        if self.use_cache:
                            try:
                                vectorstore.save_local(str(cache_path))
                                print(f"✓ Cached {doc_type} vectorstore")
                            except Exception as e:
                                print(f"Warning: Failed to cache {doc_type}: {e}")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: {filename} not found in {self.docs_path}")
    
    def query_hardware_info(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query hardware characteristics"""
        return self._query_vectorstore('hardware', query, top_k)
    
    def query_tutorials(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query Triton tutorials and standard implementations"""
        return self._query_vectorstore('tutorials', query, top_k)
    
    def query_optimization(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query optimization techniques"""
        return self._query_vectorstore('optimization', query, top_k)
    
    def query_all(self, query: str, top_k_per_type: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Query all document types"""
        results = {}
        for doc_type in self.vectorstores.keys():
            results[doc_type] = self._query_vectorstore(doc_type, query, top_k_per_type)
        return results
    
    def _query_vectorstore(self, doc_type: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query specific vectorstore"""
        if doc_type not in self.vectorstores or not self.embeddings:
            return []
        
        try:
            vectorstore = self.vectorstores[doc_type]
            docs = vectorstore.similarity_search(query, k=top_k)
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', ''),
                    'file': doc.metadata.get('file', ''),
                    'relevance_score': 1.0  # FAISS doesn't return scores by default
                })
            
            return results
            
        except Exception as e:
            print(f"Error querying {doc_type}: {e}")
            return []
    
    def get_context_for_operator(self, operator_name: str, instruction: str) -> Dict[str, str]:
        """
        Get relevant context for a specific operator with enhanced query strategy
        
        Args:
            operator_name: Name of the operator (e.g., 'matmul', 'layernorm')
            instruction: Full instruction text
            
        Returns:
            Dictionary with context from different sources
        """
        context = {
            'hardware_context': '',
            'tutorial_context': '',
            'optimization_context': ''
        }
        
        if not self.embeddings:
            return context
        
        # Extract operator type from name with enhanced mapping
        op_type = self._extract_operator_type(operator_name)
        
        # Enhanced query building with fallback
        try:
            # Query hardware info with enhanced terms
            hw_query = f"{op_type} GPU memory bandwidth occupancy shared memory tensor core warp"
            hw_results = self.query_hardware_info(hw_query, top_k=3)
            if hw_results:
                context['hardware_context'] = "\n\n".join([r['content'] for r in hw_results])
            
            # Query tutorials with multiple strategies
            tutorial_queries = [
                f"{op_type} triton implementation kernel",
                f"{op_type} example code snippet",
                f"triton {op_type} tutorial"
            ]
            
            all_tutorial_results = []
            for tq in tutorial_queries:
                results = self.query_tutorials(tq, top_k=2)
                all_tutorial_results.extend(results)
            
            # Deduplicate and take best results
            seen_content = set()
            unique_results = []
            for result in all_tutorial_results:
                content_hash = hash(result['content'][:100])  # Hash first 100 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
                    if len(unique_results) >= 3:
                        break
            
            if unique_results:
                context['tutorial_context'] = "\n\n".join([r['content'] for r in unique_results])
            
            # Query optimization techniques with enhanced terms
            opt_query = f"{op_type} optimization memory coalescing block size num_warps performance"
            opt_results = self.query_optimization(opt_query, top_k=3)
            if opt_results:
                context['optimization_context'] = "\n\n".join([r['content'] for r in opt_results])
                
        except Exception as e:
            print(f"Error in enhanced context retrieval: {e}")
            # Fallback to simple retrieval
            return self._simple_context_retrieval(op_type, instruction)
        
        return context
    
    def _simple_context_retrieval(self, op_type: str, instruction: str) -> Dict[str, str]:
        """Fallback simple context retrieval"""
        context = {
            'hardware_context': '',
            'tutorial_context': '',
            'optimization_context': ''
        }
        
        # Simple queries as fallback
        hw_results = self.query_hardware_info(f"{op_type} GPU", top_k=2)
        if hw_results:
            context['hardware_context'] = hw_results[0]['content']
            
        tutorial_results = self.query_tutorials(f"{op_type} triton", top_k=2)
        if tutorial_results:
            context['tutorial_context'] = tutorial_results[0]['content']
            
        opt_results = self.query_optimization(f"{op_type} optimization", top_k=2)
        if opt_results:
            context['optimization_context'] = opt_results[0]['content']
            
        return context
    
    def _extract_operator_type(self, operator_name: str) -> str:
        """Extract operator type from filename"""
        name = operator_name.lower().replace('.py', '')
        
        # Map common operators
        operator_mapping = {
            'matmul': 'matrix multiplication gemm',
            'layernorm': 'layer normalization',
            'softmax': 'softmax attention',
            'attention': 'attention flash attention',
            'conv': 'convolution',
            'relu': 'activation relu',
            'gelu': 'activation gelu',
            'add': 'elementwise addition',
            'transpose': 'transpose memory',
            'reduce': 'reduction sum',
            'embedding': 'embedding lookup'
        }
        
        for key, value in operator_mapping.items():
            if key in name:
                return value
        
        return name  # fallback to original name


class FallbackRAGRetriever:
    """
    Fallback retriever when LangChain is not available
    Uses simple keyword matching
    """
    
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.documents = {}
        self._load_documents()
    
    def _load_documents(self):
        """Load documents as plain text"""
        doc_files = {
            'hardware': 'hardware_characteristics.md',
            'tutorials': 'triton_tutorials.md',
            'optimization': 'optimization_techniques.md'
        }
        
        for doc_type, filename in doc_files.items():
            file_path = self.docs_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.documents[doc_type] = f.read()
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def get_context_for_operator(self, operator_name: str, instruction: str) -> Dict[str, str]:
        """Simple keyword-based context extraction"""
        context = {
            'hardware_context': '',
            'tutorial_context': '',
            'optimization_context': ''
        }
        
        op_type = operator_name.lower().replace('.py', '')
        keywords = [op_type, 'memory', 'optimization', 'block', 'kernel']
        
        for doc_type, content in self.documents.items():
            # Simple keyword matching
            relevant_sections = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in keywords):
                    # Extract surrounding context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 5)
                    section = '\n'.join(lines[start:end])
                    relevant_sections.append(section)
            
            if relevant_sections:
                context[f'{doc_type}_context'] = '\n\n'.join(relevant_sections[:2])  # Top 2 sections
        
        return context


def create_rag_retriever(docs_path: str = "docs") -> Any:
    """Factory function to create appropriate RAG retriever"""
    if LANGCHAIN_AVAILABLE:
        try:
            return TritonRAGRetriever(docs_path)
        except Exception as e:
            print(f"Failed to initialize LangChain RAG: {e}")
            print("Falling back to simple retriever")
            return FallbackRAGRetriever(docs_path)
    else:
        return FallbackRAGRetriever(docs_path)