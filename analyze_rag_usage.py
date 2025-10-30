#!/usr/bin/env python3
"""
Analyze RAG usage from generated reports
"""

import json
import sys
from pathlib import Path
import argparse

def analyze_rag_report(report_path: str):
    """Analyze and display RAG usage report"""
    
    if not Path(report_path).exists():
        print(f"❌ Report file not found: {report_path}")
        return False
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load report: {e}")
        return False
    
    print("🔍 RAG USAGE ANALYSIS")
    print("=" * 60)
    
    # Summary
    summary = report['summary']
    print(f"📊 SUMMARY:")
    print(f"   Total Operators: {summary['total_operators']}")
    print(f"   RAG-Enhanced: {summary['rag_enabled_operators']} ({summary['rag_enabled_operators']/summary['total_operators']*100:.1f}%)")
    print(f"   Total Retrievals: {summary['total_rag_retrievals']}")
    print(f"   Avg Context Length: {summary['avg_context_length']:.0f} characters")
    
    # Per-operator analysis
    print(f"\n📋 PER-OPERATOR DETAILS:")
    
    operators = report['per_operator']
    rag_users = [(name, data) for name, data in operators.items() if data['iterations_with_rag'] > 0]
    non_rag_users = [(name, data) for name, data in operators.items() if data['iterations_with_rag'] == 0]
    
    # RAG users
    if rag_users:
        print(f"\n✅ OPERATORS USING RAG ({len(rag_users)}):")
        for name, data in sorted(rag_users, key=lambda x: x[1]['iterations_with_rag'], reverse=True):
            sources = ', '.join([f"{k}({v})" for k, v in data['sources_used'].items()])
            print(f"   {name}: {data['iterations_with_rag']} iterations, {data['total_context_chars']} chars, sources: {sources}")
    
    # Non-RAG users
    if non_rag_users:
        print(f"\n❌ OPERATORS NOT USING RAG ({len(non_rag_users)}):")
        for name, data in non_rag_users:
            note = data.get('note', 'No RAG context found')
            print(f"   {name}: {note}")
    
    # Source usage statistics
    print(f"\n📚 SOURCE USAGE STATISTICS:")
    all_sources = {}
    for data in operators.values():
        for source, count in data.get('sources_used', {}).items():
            all_sources[source] = all_sources.get(source, 0) + count
    
    if all_sources:
        for source, count in sorted(all_sources.items(), key=lambda x: x[1], reverse=True):
            print(f"   {source}: {count} retrievals")
    else:
        print("   No source usage recorded")
    
    # Detailed iteration analysis
    print(f"\n🔄 ITERATION-BY-ITERATION ANALYSIS:")
    for name, data in rag_users[:3]:  # Show top 3 RAG users
        print(f"\n   📄 {name}:")
        for i, usage in enumerate(data['usage_per_iteration']):
            sources_str = ', '.join(usage['sources'])
            print(f"      Iter {usage['iteration']}: {usage['total_context_length']} chars ({sources_str})")
    
    return True

def compare_with_without_rag(rag_report_path: str, baseline_report_path: str = None):
    """Compare performance with and without RAG (if baseline available)"""
    
    # This would require baseline results without RAG
    # For now, just show RAG impact analysis
    
    print(f"\n🆚 RAG IMPACT ANALYSIS:")
    print("   (Requires baseline run without RAG for comparison)")
    print("   Current analysis shows RAG usage patterns only")

def main():
    parser = argparse.ArgumentParser(description="Analyze RAG usage reports")
    parser.add_argument("report_path", help="Path to RAG usage report JSON file")
    parser.add_argument("--baseline", help="Path to baseline report for comparison")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-iteration analysis")
    
    args = parser.parse_args()
    
    success = analyze_rag_report(args.report_path)
    
    if args.baseline:
        compare_with_without_rag(args.report_path, args.baseline)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())