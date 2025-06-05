#!/usr/bin/env python3
"""
Dataset sanity check script - verify raw data and processing functions.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

# Import dataset loading and formatting functions
from datasets import load_dataset
from dataset_runner import (
    load_medqa_dataset, load_medmcqa_dataset, load_pubmedqa_dataset, load_mmlupro_med_dataset,
    format_medqa_for_task, format_medmcqa_for_task, format_pubmedqa_for_task, format_mmlupro_med_for_task
)

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_dataset_structure(dataset_name: str, raw_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the structure of raw dataset questions."""
    if not raw_questions:
        return {"error": "No questions to analyze"}
    
    # Get all unique keys across all questions
    all_keys = set()
    for q in raw_questions:
        all_keys.update(q.keys())
    
    # Analyze field types and sample values
    field_analysis = {}
    for key in all_keys:
        values = [q.get(key) for q in raw_questions if key in q]
        non_none_values = [v for v in values if v is not None]
        
        field_analysis[key] = {
            "present_in": len([q for q in raw_questions if key in q]),
            "total_questions": len(raw_questions),
            "coverage": len([q for q in raw_questions if key in q]) / len(raw_questions),
            "types": list(set(type(v).__name__ for v in non_none_values)),
            "sample_values": non_none_values[:3] if non_none_values else [],
            "unique_count": len(set(str(v) for v in non_none_values)) if non_none_values else 0
        }
    
    return {
        "dataset": dataset_name,
        "total_questions": len(raw_questions),
        "unique_fields": len(all_keys),
        "fields": field_analysis
    }

def process_questions(dataset_name: str, raw_questions: List[Dict[str, Any]], num_questions: int) -> List[Dict[str, Any]]:
    """Process raw questions using our formatting functions."""
    processed = []
    
    format_functions = {
        "medqa": format_medqa_for_task,
        "medmcqa": format_medmcqa_for_task,
        "pubmedqa": format_pubmedqa_for_task,
        "mmlupro-med": format_mmlupro_med_for_task
    }
    
    format_func = format_functions.get(dataset_name)
    if not format_func:
        return [{"error": f"No format function for {dataset_name}"}]
    
    for i, raw_q in enumerate(raw_questions[:num_questions]):
        try:
            if dataset_name in ["medmcqa"]:  # Returns tuple
                agent_task, eval_data = format_func(raw_q)
                processed_item = {
                    "question_index": i,
                    "raw_question": raw_q,
                    "agent_task": agent_task,
                    "eval_data": eval_data,
                    "processing_success": True
                }
            else:  # Returns single dict
                try:
                    # Try tuple first
                    agent_task, eval_data = format_func(raw_q)
                    processed_item = {
                        "question_index": i,
                        "raw_question": raw_q,
                        "agent_task": agent_task,
                        "eval_data": eval_data,
                        "processing_success": True
                    }
                except ValueError:
                    # Falls back to single return
                    task = format_func(raw_q)
                    processed_item = {
                        "question_index": i,
                        "raw_question": raw_q,
                        "formatted_task": task,
                        "processing_success": True,
                        "note": "Single return format (needs GT separation fix)"
                    }
        except Exception as e:
            processed_item = {
                "question_index": i,
                "raw_question": raw_q,
                "processing_error": str(e),
                "processing_success": False
            }
        
        processed.append(processed_item)
    
    return processed

def print_summary(dataset_name: str, structure_analysis: Dict[str, Any], processed_questions: List[Dict[str, Any]]):
    """Print summary to console."""
    print(f"\n{'='*60}")
    print(f"DATASET SANITY CHECK: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Dataset structure
    print(f"\nRAW DATASET STRUCTURE:")
    print(f"  Total questions: {structure_analysis['total_questions']}")
    print(f"  Unique fields: {structure_analysis['unique_fields']}")
    
    # Field coverage
    print(f"\nFIELD COVERAGE:")
    for field, info in sorted(structure_analysis['fields'].items()):
        coverage = info['coverage'] * 100
        print(f"  {field:20} | {coverage:5.1f}% | {info['types']} | Sample: {info['sample_values']}")
    
    # Processing results
    success_count = sum(1 for q in processed_questions if q.get('processing_success', False))
    error_count = len(processed_questions) - success_count
    
    print(f"\nPROCESSING RESULTS:")
    print(f"  Successful: {success_count}/{len(processed_questions)}")
    print(f"  Errors: {error_count}/{len(processed_questions)}")
    
    if error_count > 0:
        print(f"\nERRORS:")
        for q in processed_questions:
            if not q.get('processing_success', False):
                print(f"  Q{q['question_index']}: {q.get('processing_error', 'Unknown error')}")
    
    # GT leak check
    gt_leak_detected = False
    for q in processed_questions:
        if q.get('processing_success', False):
            agent_task = q.get('agent_task', q.get('formatted_task', {}))
            if isinstance(agent_task, dict) and 'ground_truth' in agent_task:
                gt_leak_detected = True
                break
    
    print(f"\nGROUND TRUTH LEAK CHECK:")
    print(f"  GT in agent task: {'⚠️  YES - LEAK DETECTED!' if gt_leak_detected else '✅ NO - Safe'}")

def save_results(output_dir: str, dataset_name: str, structure_analysis: Dict[str, Any], processed_questions: List[Dict[str, Any]]):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save structure analysis
    structure_file = os.path.join(output_dir, f"{dataset_name}_structure_{timestamp}.json")
    with open(structure_file, 'w') as f:
        json.dump(structure_analysis, f, indent=2, default=str)
    
    # Save processed questions
    processed_file = os.path.join(output_dir, f"{dataset_name}_processed_{timestamp}.json")
    with open(processed_file, 'w') as f:
        json.dump(processed_questions, f, indent=2, default=str)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"{dataset_name}_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset Sanity Check Report: {dataset_name}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        f.write("RAW DATASET FIELDS:\n")
        for field, info in sorted(structure_analysis['fields'].items()):
            f.write(f"  {field}: {info['coverage']*100:.1f}% coverage, types: {info['types']}\n")
        
        f.write(f"\nPROCESSING STATUS:\n")
        success_count = sum(1 for q in processed_questions if q.get('processing_success', False))
        f.write(f"  Success: {success_count}/{len(processed_questions)}\n")
        f.write(f"  Errors: {len(processed_questions) - success_count}/{len(processed_questions)}\n")
        
        # GT leak check
        gt_leak = any('ground_truth' in q.get('agent_task', q.get('formatted_task', {})) 
                     for q in processed_questions if q.get('processing_success', False))
        f.write(f"\nGROUND TRUTH LEAK: {'DETECTED' if gt_leak else 'NOT DETECTED'}\n")
    
    print(f"\nFiles saved to {output_dir}:")
    print(f"  Structure: {structure_file}")
    print(f"  Processed: {processed_file}")
    print(f"  Summary: {summary_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Dataset sanity check - analyze raw and processed data')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['medqa', 'medmcqa', 'pubmedqa', 'mmlupro-med'],
                      help='Dataset to analyze')
    parser.add_argument('--n', type=int, default=10,
                      help='Number of questions to analyze (default: 10)')
    parser.add_argument('--output-dir', type=str, default='sanity_check_results',
                      help='Output directory for results (default: sanity_check_results)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print(f"Starting sanity check for {args.dataset} dataset...")
    print(f"Analyzing {args.n} questions...")
    
    # Load raw dataset
    load_functions = {
        "medqa": load_medqa_dataset,
        "medmcqa": load_medmcqa_dataset,
        "pubmedqa": load_pubmedqa_dataset,
        "mmlupro-med": load_mmlupro_med_dataset
    }
    
    load_func = load_functions[args.dataset]
    raw_questions = load_func(num_questions=args.n, random_seed=args.seed)
    
    if not raw_questions:
        print(f"❌ Failed to load {args.dataset} dataset")
        return
    
    print(f"✅ Loaded {len(raw_questions)} questions from {args.dataset}")
    
    # Analyze structure
    structure_analysis = analyze_dataset_structure(args.dataset, raw_questions)
    
    # Process questions
    processed_questions = process_questions(args.dataset, raw_questions, args.n)
    
    # Print summary
    print_summary(args.dataset, structure_analysis, processed_questions)
    
    # Save results
    save_results(args.output_dir, args.dataset, structure_analysis, processed_questions)
    
    print(f"\n✅ Sanity check completed for {args.dataset}")

if __name__ == "__main__":
    main()




"""
# Check MedQA with 5 questions
python sanity.py --dataset medqa --n 5

# Check MedMCQA with 10 questions, custom output
python sanity.py --dataset medmcqa --n 10 --output-dir my_sanity_results

# Check all datasets
python sanity.py --dataset medqa --n 3
python sanity.py --dataset medmcqa --n 3  
python sanity.py --dataset pubmedqa --n 3
python sanity.py --dataset mmlupro-med --n 3

"""