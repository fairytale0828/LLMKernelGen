#!/usr/bin/env python3
"""
RAG Diagnostics and Monitoring Tool
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse

def check_rag_setup(docs_path: str = "docs") -> Dict[str, Any]:
    """Check RAG setup and configuration"""
    
    results = {
        'status': 'ok',
        'issues': [],
        'warnings': [],
        'info': []
    }
    
    docs_dir = Path(docs_path)
    
    # Check if docs directory exists
    if not docs_dir.exists():
        results['issues'].append(f"Documentation directory not found: {docs_path}")
        results['status'] = 'error'
        return results
    
    # Check required documents
    required_docs = [
        'hardware_characteristics.md',
        'triton_tutorials.md', 
        'optimization_techniques.md'
    ]
    
    for doc in required_docs:
        doc_path = docs_dir / doc
        if not doc_path.exists():
            results['issues'].append(f"Required document missing: {doc}")
            results['status'] = 'error'
        else:
            # Check document size
            size = doc_path.stat().st_size
            if size < 1000:  # Less than 1KB
                results['warnings'].append(f"Document {doc} is very small ({size} bytes)")
            else:
                results['info'].append(f"✓ {doc}: {size:,} bytes")
    
    # Check dependencies
    try:
        import langchain
        results['info'].append(f"✓ LangChain version: {langchain.__version__}")
    except ImportError:
        results['issues'].append("LangChain not installed")
        results['status'] = 'error'
    
    try:
        import faiss
        results['info'].append("✓ FAISS available")
    except ImportError:
        results['issues'].append("FAISS not installed")
        results['status'] = 'error'
    
    try:
        import sentence_transformers
        results['info'].append(f"✓ Sentence Transformers available")
    except ImportError:
        results['issues'].append("Sentence Transformers not installed")
        results['status'] = 'error'
    
    # Check cache directory
    cache_dir = Path('.rag_cache')
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        results['info'].append(f"✓ RAG cache: {cache_size:,} bytes")
    else:
        results['warnings'].append("No RAG cache found (will be created on first run)")
    
    return results

def test_rag_retrieval(docs_path: str = "docs") -> Dict[str, Any]:
    """Test RAG retrieval functionality"""
    
    results = {
        'status': 'ok',
        'tests': [],
        'errors': []
    }
    
    try:
        sys.path.append('src')
        from retrievers.langchain_rag import create_rag_retriever
        
        # Create retriever
        retriever = create_rag_retriever(docs_path)
        
        # Test cases
        test_cases = [
            ("matmul_triton.py", "matrix multiplication"),
            ("layernorm_fwd.py", "layer normalization"),
            ("softmax_kernel.py", "softmax activation"),
            ("attention_fwd.py", "attention mechanism"),
            ("unknown_op.py", "unknown operation")
        ]
        
        for op_name, instruction in test_cases:
            try:
                context = retriever.get_context_for_operator(op_name, instruction)
                
                test_result = {
                    'operator': op_name,
                    'instruction': instruction,
                    'hardware_chars': len(context.get('hardware_context', '')),
                    'tutorial_chars': len(context.get('tutorial_context', '')),
                    'optimization_chars': len(context.get('optimization_context', '')),
                    'total_chars': sum(len(context.get(k, '')) for k in context.keys()),
                    'status': 'ok'
                }
                
                # Check if we got reasonable context
                if test_result['total_chars'] < 100:
                    test_result['status'] = 'warning'
                    test_result['note'] = 'Very little context retrieved'
                elif test_result['total_chars'] > 5000:
                    test_result['status'] = 'warning'
                    test_result['note'] = 'Very large context retrieved'
                
                results['tests'].append(test_result)
                
            except Exception as e:
                results['errors'].append(f"Error testing {op_name}: {str(e)}")
                results['status'] = 'error'
                
    except Exception as e:
        results['errors'].append(f"Failed to create RAG retriever: {str(e)}")
        results['status'] = 'error'
    
    return results

def analyze_rag_performance(usage_report_path: str) -> Dict[str, Any]:
    """Analyze RAG performance from usage report"""
    
    if not Path(usage_report_path).exists():
        return {
            'status': 'error',
            'error': f"Usage report not found: {usage_report_path}"
        }
    
    try:
        with open(usage_report_path, 'r') as f:
            report = json.load(f)
    except Exception as e:
        return {
            'status': 'error', 
            'error': f"Failed to load report: {str(e)}"
        }
    
    analysis = {
        'status': 'ok',
        'summary': {},
        'recommendations': [],
        'performance_metrics': {}
    }
    
    # Basic metrics
    summary = report.get('summary', {})
    total_ops = summary.get('total_operators', 0)
    rag_enabled = summary.get('rag_enabled_operators', 0)
    
    analysis['summary'] = {
        'total_operators': total_ops,
        'rag_enabled_operators': rag_enabled,
        'rag_adoption_rate': (rag_enabled / total_ops * 100) if total_ops > 0 else 0,
        'total_retrievals': summary.get('total_rag_retrievals', 0),
        'avg_context_length': summary.get('avg_context_length', 0)
    }
    
    # Performance analysis
    per_operator = report.get('per_operator', {})
    
    # Find operators with no RAG usage
    no_rag_ops = [name for name, data in per_operator.items() 
                  if data.get('iterations_with_rag', 0) == 0]
    
    if no_rag_ops:
        analysis['recommendations'].append({
            'type': 'coverage',
            'message': f"{len(no_rag_ops)} operators not using RAG",
            'operators': no_rag_ops[:5],  # Show first 5
            'action': 'Check if these operators need better keyword mapping'
        })
    
    # Find operators with very high context usage
    high_context_ops = []
    for name, data in per_operator.items():
        avg_context = (data.get('total_context_chars', 0) / 
                      max(data.get('iterations_with_rag', 1), 1))
        if avg_context > 3000:
            high_context_ops.append((name, avg_context))
    
    if high_context_ops:
        analysis['recommendations'].append({
            'type': 'efficiency',
            'message': f"{len(high_context_ops)} operators using very large contexts",
            'operators': [f"{name} ({chars:.0f} chars)" for name, chars in high_context_ops[:3]],
            'action': 'Consider reducing context length limits'
        })
    
    # Source usage analysis
    source_usage = {}
    for data in per_operator.values():
        for source, count in data.get('sources_used', {}).items():
            source_usage[source] = source_usage.get(source, 0) + count
    
    analysis['performance_metrics']['source_usage'] = source_usage
    
    # Check for balanced source usage
    if source_usage:
        max_usage = max(source_usage.values())
        min_usage = min(source_usage.values())
        if max_usage > min_usage * 3:  # Imbalanced usage
            analysis['recommendations'].append({
                'type': 'balance',
                'message': 'Imbalanced source usage detected',
                'details': source_usage,
                'action': 'Some document types may need better content or indexing'
            })
    
    return analysis

def generate_rag_report(docs_path: str = "docs", output_path: str = "rag_diagnostic_report.json"):
    """Generate comprehensive RAG diagnostic report"""
    
    print("🔍 Running RAG Diagnostics...")
    
    report = {
        'timestamp': str(Path().cwd()),
        'setup_check': check_rag_setup(docs_path),
        'retrieval_test': test_rag_retrieval(docs_path)
    }
    
    # Check for existing usage reports
    usage_reports = list(Path('.').glob('*_rag_usage.json'))
    if usage_reports:
        latest_report = max(usage_reports, key=lambda p: p.stat().st_mtime)
        print(f"📊 Analyzing usage report: {latest_report}")
        report['performance_analysis'] = analyze_rag_performance(str(latest_report))
    else:
        report['performance_analysis'] = {
            'status': 'info',
            'message': 'No usage reports found. Run OptimAgent to generate usage data.'
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Diagnostic report saved to: {output_path}")
    
    # Print summary
    print_diagnostic_summary(report)

def print_diagnostic_summary(report: Dict[str, Any]):
    """Print diagnostic summary to console"""
    
    print("\n" + "="*60)
    print("🔍 RAG DIAGNOSTIC SUMMARY")
    print("="*60)
    
    # Setup status
    setup = report['setup_check']
    if setup['status'] == 'ok':
        print("✅ Setup: OK")
    else:
        print(f"❌ Setup: {setup['status'].upper()}")
        for issue in setup['issues']:
            print(f"   • {issue}")
    
    for info in setup['info']:
        print(f"   {info}")
    
    for warning in setup['warnings']:
        print(f"   ⚠️  {warning}")
    
    # Retrieval test status
    retrieval = report['retrieval_test']
    if retrieval['status'] == 'ok':
        print(f"\n✅ Retrieval Test: OK ({len(retrieval['tests'])} tests)")
        
        # Show test results
        for test in retrieval['tests']:
            status_icon = "✅" if test['status'] == 'ok' else "⚠️"
            print(f"   {status_icon} {test['operator']}: {test['total_chars']} chars total")
    else:
        print(f"\n❌ Retrieval Test: FAILED")
        for error in retrieval['errors']:
            print(f"   • {error}")
    
    # Performance analysis
    perf = report.get('performance_analysis', {})
    if perf.get('status') == 'ok':
        summary = perf['summary']
        print(f"\n📊 Performance Analysis:")
        print(f"   • RAG Adoption: {summary['rag_adoption_rate']:.1f}% ({summary['rag_enabled_operators']}/{summary['total_operators']})")
        print(f"   • Total Retrievals: {summary['total_retrievals']}")
        print(f"   • Avg Context Length: {summary['avg_context_length']:.0f} chars")
        
        # Show recommendations
        for rec in perf.get('recommendations', []):
            print(f"   💡 {rec['message']}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="RAG Diagnostics Tool")
    parser.add_argument("--docs-path", default="docs", help="Path to documentation directory")
    parser.add_argument("--output", default="rag_diagnostic_report.json", help="Output report path")
    parser.add_argument("--usage-report", help="Path to existing usage report for analysis")
    parser.add_argument("--setup-only", action="store_true", help="Only check setup")
    parser.add_argument("--test-only", action="store_true", help="Only test retrieval")
    
    args = parser.parse_args()
    
    if args.setup_only:
        result = check_rag_setup(args.docs_path)
        print(json.dumps(result, indent=2))
    elif args.test_only:
        result = test_rag_retrieval(args.docs_path)
        print(json.dumps(result, indent=2))
    elif args.usage_report:
        result = analyze_rag_performance(args.usage_report)
        print(json.dumps(result, indent=2))
    else:
        generate_rag_report(args.docs_path, args.output)

if __name__ == "__main__":
    main()