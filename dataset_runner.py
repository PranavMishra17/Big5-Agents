"""
Dataset runner for the modular agent system.
Runs multiple questions from datasets through the agent system.
"""

import argparse
import logging
import os
import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm

from datasets import load_dataset
from simulator import AgentSystemSimulator
import config
from logger import SimulationLogger

def setup_logging():
    """Set up logging for the dataset runner."""
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, "dataset_runner.log")),
            logging.StreamHandler()
        ]
    )

def load_medqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load questions from the MedQA dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MedQA dataset with {num_questions} random questions")
    
    # Load the dataset
    try:
        ds = load_dataset("sickgpt/001_MedQA_raw")
        
        # Convert to list for easier processing
        questions = list(ds["train"])
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedQA dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedQA dataset: {str(e)}")
        return []

def load_pubmedqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load questions from the PubMedQA dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading PubMedQA dataset with {num_questions} random questions")
    
    # Load the dataset
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        
        # Convert to list for easier processing
        questions = list(ds["train"])
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from PubMedQA dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading PubMedQA dataset: {str(e)}")
        return []

def format_medqa_for_task(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format MedQA question for the agent system task.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Task dictionary for config.TASK
    """
    # Extract question and choices
    question_text = question_data.get("question", "")
    choices = question_data.get("choices", [])
    
    # Ensure choices is a list (sometimes it might be nested differently)
    if not isinstance(choices, list):
        choices = []
    
    # Format choices as options
    options = []
    for i, choice in enumerate(choices):
        if isinstance(choice, str):
            options.append(f"{chr(65+i)}. {choice}")  # A, B, C, D, etc.
    
    # Get the expected output (correct answer)
    expected_output = question_data.get("expected_output", "")
    
    # Create task dictionary
    task = {
        "name": "MedQA Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale",
        "ground_truth": expected_output,
        "rationale": {}  # No rationale provided in the dataset
    }
    
    return task

def format_pubmedqa_for_task(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format PubMedQA question for the agent system task.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Task dictionary for config.TASK
    """
    # Extract question and context
    question_text = question_data.get("question", "")
    context = question_data.get("context", {})
    
    # Format context as a string
    context_text = ""
    if isinstance(context, dict):
        for i, (section, text) in enumerate(context.items()):
            context_text += f"{section}: {text}\n\n"
    elif isinstance(context, str):
        context_text = context
    
    # Get the expected output (correct answer)
    expected_output = question_data.get("final_decision", "")
    
    # Create task dictionary
    task = {
        "name": "PubMedQA Question",
        "description": f"Question: {question_text}\n\nContext: {context_text}",
        "type": "decision_with_explanation",
        "options": ["yes", "no", "maybe"],
        "expected_output_format": "Decision (yes/no/maybe) with detailed justification",
        "ground_truth": expected_output,
        "rationale": {"long_answer": question_data.get("long_answer", "")}
    }
    
    return task

def run_questions_with_configuration(
    questions: List[Dict[str, Any]],
    dataset_type: str,
    configuration: Dict[str, bool],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a set of questions with a specific configuration.
    
    Args:
        questions: List of questions to run
        dataset_type: Type of dataset ("medqa" or "pubmedqa")
        configuration: Configuration of teamwork components
        output_dir: Optional output directory for results
        
    Returns:
        Results dictionary
    """
    config_name = configuration.get("name", "unknown")
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
    # Setup output directory
    if output_dir:
        run_output_dir = os.path.join(output_dir, f"{dataset_type}_{config_name.lower().replace(' ', '_')}")
        os.makedirs(run_output_dir, exist_ok=True)
    else:
        run_output_dir = None
    
    # Results collection
    results = {
        "configuration": config_name,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "question_results": [],
        "summary": {
            "majority_voting": {"correct": 0, "total": 0},
            "weighted_voting": {"correct": 0, "total": 0},
            "borda_count": {"correct": 0, "total": 0}
        }
    }
    
    # Process each question
    for i, question in enumerate(tqdm(questions, desc=f"{config_name}")):
        # Format the question for the task
        if dataset_type == "medqa":
            task = format_medqa_for_task(question)
        elif dataset_type == "pubmedqa":
            task = format_pubmedqa_for_task(question)
        else:
            logging.error(f"Unknown dataset type: {dataset_type}")
            continue
        
        # Update the global task configuration
        config.TASK = task
        
        # Create simulator with specified configuration
        simulator = AgentSystemSimulator(
            simulation_id=f"{dataset_type}_{config_name.lower().replace(' ', '_')}_{i}",
            use_team_leadership=configuration.get("leadership", False),
            use_closed_loop_comm=configuration.get("closed_loop", False),
            use_mutual_monitoring=configuration.get("mutual_monitoring", False),
            use_shared_mental_model=configuration.get("shared_mental_model", False)
        )
        
        try:
            # Run the simulation
            simulation_results = simulator.run_simulation()
            
            # Evaluate performance
            performance = simulator.evaluate_performance()
            
            # Combine results
            question_result = {
                "question_index": i,
                "question": task["description"],
                "ground_truth": task.get("ground_truth", ""),
                "decisions": simulation_results["decision_results"],
                "performance": performance.get("task_performance", {})
            }
            
            # Update summary statistics
            for method in ["majority_voting", "weighted_voting", "borda_count"]:
                if method in performance.get("task_performance", {}):
                    method_perf = performance["task_performance"][method]
                    if "correct" in method_perf:
                        results["summary"][method]["total"] += 1
                        if method_perf["correct"]:
                            results["summary"][method]["correct"] += 1
            
            # Save results
            results["question_results"].append(question_result)
            
            # Save to specific output directory if provided
            if run_output_dir:
                with open(os.path.join(run_output_dir, f"question_{i}.json"), 'w') as f:
                    json.dump({
                        "question": task["description"],
                        "options": task.get("options", []),
                        "ground_truth": task.get("ground_truth", ""),
                        "decisions": simulation_results["decision_results"],
                        "performance": performance.get("task_performance", {})
                    }, f, indent=2)
            
        except Exception as e:
            logging.error(f"Error running question {i}: {str(e)}")
            results["question_results"].append({
                "question_index": i,
                "question": task["description"],
                "error": str(e)
            })
    
    # Calculate accuracy for each method
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        method_summary = results["summary"][method]
        if method_summary["total"] > 0:
            method_summary["accuracy"] = method_summary["correct"] / method_summary["total"]
        else:
            method_summary["accuracy"] = 0.0
    
    # Save overall results
    if run_output_dir:
        with open(os.path.join(run_output_dir, "summary.json"), 'w') as f:
            json.dump(results["summary"], f, indent=2)
    
    # Print summary
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
    
    return results

def run_dataset(
    dataset_type: str,
    num_questions: int = 50,
    random_seed: int = 42,
    run_all_configs: bool = False,
    output_dir: Optional[str] = None,
    leadership: bool = False,
    closed_loop: bool = False,
    mutual_monitoring: bool = False,
    shared_mental_model: bool = False
) -> Dict[str, Any]:
    """
    Run a dataset through the agent system.
    
    Args:
        dataset_type: Type of dataset ("medqa" or "pubmedqa")
        num_questions: Number of questions to process
        random_seed: Random seed for reproducibility
        run_all_configs: Whether to run all configurations
        output_dir: Optional output directory for results
        leadership: Use team leadership
        closed_loop: Use closed-loop communication
        mutual_monitoring: Use mutual performance monitoring
        shared_mental_model: Use shared mental model
        
    Returns:
        Results dictionary
    """
    # Load the dataset
    if dataset_type == "medqa":
        questions = load_medqa_dataset(num_questions, random_seed)
    elif dataset_type == "pubmedqa":
        questions = load_pubmedqa_dataset(num_questions, random_seed)
    else:
        logging.error(f"Unknown dataset type: {dataset_type}")
        return {"error": f"Unknown dataset type: {dataset_type}"}
    
    if not questions:
        return {"error": "No questions loaded"}
    
    # Setup output directory
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"{dataset_type}_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
    else:
        run_output_dir = None
    
    # Define configurations to test
    if run_all_configs:
        configurations = [
            # Baseline (no features)
            {
                "name": "Baseline", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False
            },
            # Single features
            {
                "name": "Leadership", 
                "leadership": True, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False
            },
            {
                "name": "Closed-loop", 
                "leadership": False, 
                "closed_loop": True,
                "mutual_monitoring": False,
                "shared_mental_model": False
            },
            {
                "name": "Mutual Monitoring", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": True,
                "shared_mental_model": False
            },
            {
                "name": "Shared Mental Model", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": True
            },
            # All features
            {
                "name": "All Features", 
                "leadership": True, 
                "closed_loop": True,
                "mutual_monitoring": True,
                "shared_mental_model": True
            }
        ]
    else:
        # Use the specified configuration
        configurations = [{
            "name": "Custom Configuration",
            "leadership": leadership,
            "closed_loop": closed_loop,
            "mutual_monitoring": mutual_monitoring,
            "shared_mental_model": shared_mental_model
        }]
    
    # Run each configuration
    all_results = []
    for config in configurations:
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config,
            run_output_dir
        )
        all_results.append(result)
    
    # Compile combined results
    combined_results = {
        "dataset": dataset_type,
        "num_questions": num_questions,
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
        "configurations": [r["configuration"] for r in all_results],
        "summaries": {r["configuration"]: r["summary"] for r in all_results}
    }
    
    # Save combined results
    if run_output_dir:
        with open(os.path.join(run_output_dir, "combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
    
    return combined_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run datasets through the agent system')
    parser.add_argument('--dataset', type=str, default="medqa", choices=["medqa", "pubmedqa"], 
                      help='Dataset to run (medqa or pubmedqa)')
    parser.add_argument('--num-questions', type=int, default=50, 
                      help='Number of questions to process')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--all', action='store_true', 
                      help='Run all feature configurations')
    parser.add_argument('--output-dir', type=str, default=None, 
                      help='Output directory for results')
    
    # Teamwork components
    parser.add_argument('--leadership', action='store_true', 
                      help='Use team leadership')
    parser.add_argument('--closedloop', action='store_true', 
                      help='Use closed-loop communication')
    parser.add_argument('--mutual', action='store_true', 
                      help='Use mutual performance monitoring')
    parser.add_argument('--mental', action='store_true', 
                      help='Use shared mental model')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Run dataset
    results = run_dataset(
        dataset_type=args.dataset,
        num_questions=args.num_questions,
        random_seed=args.seed,
        run_all_configs=args.all,
        output_dir=args.output_dir or config.OUTPUT_DIR,
        leadership=args.leadership,
        closed_loop=args.closedloop,
        mutual_monitoring=args.mutual,
        shared_mental_model=args.mental
    )
    
    # Print overall summary
    print("\nOverall Results:")
    for config_name, summary in results.get("summaries", {}).items():
        print(f"\n{config_name}:")
        for method, stats in summary.items():
            if "accuracy" in stats:
                print(f"  {method.replace('_', ' ').title()}: {stats['accuracy']:.2%} accuracy")

if __name__ == "__main__":
    main()


# Run 50 random MedQA questions with all teamwork components
#python dataset_runner.py --dataset medqa --num-questions 50 --leadership --closedloop --mutual --mental

# Run all configurations (baseline, individual components, all components) on PubMedQA
#python dataset_runner.py --dataset pubmedqa --num-questions 25 --all

# Specify custom output directory and random seed
#python dataset_runner.py --dataset medqa --output-dir ./results --seed 123 --all

# python dataset_runner.py --dataset medqa --output-dir ./medqa_results --seed 123 --num-questions 50 --leadership --closedloop --mutual --mental --all