"""
Dataset runner for the modular agent system.
Runs multiple questions from datasets through the agent system.
"""

import argparse
import logging
import os
import json
import random
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from tqdm import tqdm

from datasets import load_dataset
from simulator import AgentSystemSimulator
import config
from utils.logger import SimulationLogger

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
    output_dir: Optional[str] = None,
    max_retries: int = 3,
    n_max: int = 5
) -> Dict[str, Any]:
    """
    Run questions with specific configuration, handling errors gracefully.
    """
    config_name = configuration.get("name", "unknown")
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
    from components import agent_recruitment
    agent_recruitment.reset_complexity_metrics()

    # Setup output directory
    run_output_dir = os.path.join(output_dir, f"{dataset_type}_{config_name.lower().replace(' ', '_')}") if output_dir else None
    if run_output_dir:
        os.makedirs(run_output_dir, exist_ok=True)
    
    # Results collection
    results = {
        "configuration": config_name,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "question_results": [],
        "errors": [],
        "summary": {
            "majority_voting": {"correct": 0, "total": 0},
            "weighted_voting": {"correct": 0, "total": 0},
            "borda_count": {"correct": 0, "total": 0}
        }
    }
    
    # Process each question
    for i, question in enumerate(tqdm(questions, desc=f"{config_name}")):
        question_result = {"question_index": i}
        simulator = None  # Initialize simulator variable
        performance = None  # Initialize performance variable
        
        try:
            # Format the question for the task
            if dataset_type == "medqa":
                task = format_medqa_for_task(question)
            elif dataset_type == "pubmedqa":
                task = format_pubmedqa_for_task(question)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            question_result["question"] = task["description"]
            question_result["ground_truth"] = task.get("ground_truth", "")
            
            # Update task configuration
            config.TASK = task
            
            # Try to run the simulation with retries
            for attempt in range(max_retries):
                try:
                    if simulator is not None and hasattr(simulator, "metadata") and "complexity" in simulator.metadata:
                        complexity = simulator.metadata["complexity"]
                    # Create simulator
                    simulator = AgentSystemSimulator(
                        simulation_id=f"{dataset_type}_{config_name.lower().replace(' ', '_')}_{i}",
                        use_team_leadership=configuration.get("leadership", False),
                        use_closed_loop_comm=configuration.get("closed_loop", False),
                        use_mutual_monitoring=configuration.get("mutual_monitoring", False),
                        use_shared_mental_model=configuration.get("shared_mental_model", False),
                        use_recruitment=configuration.get("recruitment", False),
                        recruitment_method=configuration.get("recruitment_method", "adaptive"),
                        recruitment_pool=configuration.get("recruitment_pool", "general"),
                        n_max=n_max if n_max is not None else 5
                    )
                    
                    # Run simulation
                    simulation_results = simulator.run_simulation()
                    performance = simulator.evaluate_performance()
                    
                    # Store decisions and performance
                    question_result["decisions"] = simulation_results["decision_results"]
                    question_result["performance"] = performance.get("task_performance", {})
                    
                    if performance is not None:
                        # Update summary statistics
                        for method in ["majority_voting", "weighted_voting", "borda_count"]:
                            if method in performance.get("task_performance", {}):
                                method_perf = performance["task_performance"][method]
                                if "correct" in method_perf:
                                    results["summary"][method]["total"] += 1
                                    if method_perf["correct"]:
                                        results["summary"][method]["correct"] += 1
                    
                    # Simulation succeeded, break the retry loop
                    break
                
                except Exception as e:
                    error_str = str(e)
                    
                    # Check for different error types
                    if attempt < max_retries - 1:
                        # Content filter errors
                        if "content" in error_str.lower() and "filter" in error_str.lower():
                            error_type = "content_filter"
                            wait_time = 2
                            logging.warning(f"Content filter triggered, retry {attempt+1} for question {i}")
                        
                        # Rate limit errors
                        elif any(term in error_str.lower() for term in ["rate", "limit", "timeout", "capacity"]):
                            error_type = "rate_limit"
                            wait_time = min(2 ** attempt + 1, 15)  # Exponential backoff
                            logging.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}")
                        
                        # Other retryable errors
                        elif any(term in error_str.lower() for term in ["connection", "timeout", "retry", "try again"]):
                            error_type = "connection"
                            wait_time = min(2 ** attempt + 1, 10)
                            logging.warning(f"Connection error, retry {attempt+1} for question {i}")
                        
                        # Continue with retry
                        else:
                            error_type = "other"
                            wait_time = 1
                            logging.warning(f"Unknown error, retry {attempt+1} for question {i}")
                        
                        time.sleep(wait_time)
                        continue
                    
                    # Last attempt failed, record error
                    question_result["error"] = f"API error: {error_str}"
                    results["errors"].append({
                        "question_index": i,
                        "error_type": "api_error",
                        "error": error_str
                    })
                    logging.error(f"Failed after {max_retries} retries for question {i}: {error_str}")
                    break
            
            # Save individual question results
            if run_output_dir:
                try:
                    with open(os.path.join(run_output_dir, f"question_{i}.json"), 'w') as f:
                        output_data = {
                            "question": task["description"],
                            "options": task.get("options", []),
                            "ground_truth": task.get("ground_truth", "")
                        }
                        if "decisions" in question_result:
                            output_data["decisions"] = question_result["decisions"]
                        if "performance" in question_result:
                            output_data["performance"] = question_result["performance"]
                        if "error" in question_result:
                            output_data["error"] = question_result["error"]
                        json.dump(output_data, f, indent=2)
                except Exception as e:
                    logging.error(f"Failed to save results for question {i}: {str(e)}")
        
        except Exception as e:
            # Errors in task formatting or other pre-simulation errors
            logging.error(f"Error processing question {i}: {str(e)}")
            question_result["error"] = f"Processing error: {str(e)}"
            results["errors"].append({
                "question_index": i,
                "error_type": "processing",
                "error": str(e)
            })
        
        # Always add the question result, even if it has errors
        results["question_results"].append(question_result)
    
    # Calculate accuracy for each method
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        method_summary = results["summary"][method]
        method_summary["accuracy"] = method_summary["correct"] / method_summary["total"] if method_summary["total"] > 0 else 0.0

    # After simulation results and performance are obtained
    if hasattr(simulator, "metadata") and "complexity" in simulator.metadata:
        complexity = simulator.metadata["complexity"]
        # Track if this complexity level got the answer correct
        for method in ["majority_voting", "weighted_voting", "borda_count"]:
            if method in performance.get("task_performance", {}) and "correct" in performance["task_performance"][method]:
                if performance["task_performance"][method]["correct"]:
                    from components import agent_recruitment
                    if hasattr(agent_recruitment, "complexity_correct"):
                        agent_recruitment.complexity_correct[complexity] += 1
                    break  # Just count once if any method got it right
    
    # Save overall results
    if run_output_dir:
        try:
            with open(os.path.join(run_output_dir, "summary.json"), 'w') as f:
                json.dump(results["summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "errors.json"), 'w') as f:
                json.dump(results["errors"], f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save summary results: {str(e)}")
    
    # Print summary
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
    
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")

    # Add before returning results
    from components import agent_recruitment
    if hasattr(agent_recruitment, "complexity_counts") and hasattr(agent_recruitment, "complexity_correct"):
        results["complexity_metrics"] = {
            "counts": agent_recruitment.complexity_counts.copy(),
            "correct": agent_recruitment.complexity_correct.copy(),
            "accuracy": {}
        }
        
        # Calculate accuracy for each complexity level
        for level in ["basic", "intermediate", "advanced"]:
            count = results["complexity_metrics"]["counts"].get(level, 0)
            correct = results["complexity_metrics"]["correct"].get(level, 0)
            results["complexity_metrics"]["accuracy"][level] = correct / count if count > 0 else 0.0

        
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
    shared_mental_model: bool = False,
    team_orientation: bool = False,
    mutual_trust: bool = False,
    mutual_trust_factor: float = 0.8,
    recruitment: bool = False,
    recruitment_method: str = "adaptive",
    recruitment_pool: str = "general",
    n_max: int = None
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
        team_orientation: Use team orientation
        mutual_trust: Use mutual trust
        mutual_trust_factor: Mutual trust factor (0.0-1.0)
        recruitment: Use dynamic agent recruitment
        recruitment_method: Method for recruitment (adaptive, basic, intermediate, advanced)
        recruitment_pool: Pool of agent roles to recruit from
        
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
            # Baseline (no features)
            {
                "name": "Baseline", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": False
            },
            # Basic recruitment (1 agent)
            {
                "name": "Basic Recruitment", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "basic"
            },
            # Single features
            {
                "name": "Leadership", 
                "leadership": True, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            {
                "name": "Closed-loop", 
                "leadership": False, 
                "closed_loop": True,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            {
                "name": "Mutual Monitoring", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": True,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            {
                "name": "Shared Mental Model", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": True,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            {
                "name": "Team Orientation", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": True,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            {
                "name": "Mutual Trust", 
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            },
            # All features with recruitment
            {
                "name": "All Features with Recruitment", 
                "leadership": True, 
                "closed_loop": True,
                "mutual_monitoring": True,
                "shared_mental_model": True,
                "team_orientation": True,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate" if n_max is not None else "adaptive",
                "n_max": n_max
            }
        ]
    else:
        # Use the specified configuration
        configurations = [{
            "name": "Custom Configuration",
            "leadership": leadership,
            "closed_loop": closed_loop,
            "mutual_monitoring": mutual_monitoring,
            "shared_mental_model": shared_mental_model,
            "team_orientation": team_orientation,
            "mutual_trust": mutual_trust,
            "recruitment": recruitment,            
            "recruitment_method": recruitment_method,
            "n_max": n_max
        }]
    
    # Run each configuration
    all_results = []
    for config in configurations:
        # Add recruitment settings to all configs
        if config["recruitment"]:
            config["recruitment_method"] = recruitment_method
            config["recruitment_pool"] = recruitment_pool
        
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config,
            run_output_dir,
            n_max=config.get("n_max")
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
    parser.add_argument('--orientation', action='store_true', 
                      help='Use team orientation')
    parser.add_argument('--trust', action='store_true', 
                      help='Use mutual trust')
    parser.add_argument('--trust-factor', type=float, default=0.8, 
                      help='Mutual trust factor (0.0-1.0)')
    
    # Recruitment arguments
    parser.add_argument('--recruitment', action='store_true', 
                      help='Use dynamic agent recruitment')
    parser.add_argument('--recruitment-method', type=str, 
                      choices=['adaptive', 'basic', 'intermediate', 'advanced'], 
                      default='adaptive', 
                      help='Recruitment method to use')
    parser.add_argument('--recruitment-pool', type=str, 
                      choices=['general', 'medical'], 
                      default='medical' if '--dataset' in sys.argv and sys.argv[sys.argv.index('--dataset')+1] in ['medqa', 'pubmedqa'] else 'general', 
                      help='Pool of roles to recruit from')
    parser.add_argument('--n-max', type=int, default=None, 
                      help='Maximum number of agents for intermediate team')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()

    # If n_max is specified, automatically set recruitment method to intermediate
    if args.n_max is not None:
        args.recruitment = True
        args.recruitment_method = "intermediate"
    
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
        shared_mental_model=args.mental,
        team_orientation=args.orientation,
        mutual_trust=args.trust,
        mutual_trust_factor=args.trust_factor,
        recruitment=args.recruitment,
        recruitment_method=args.recruitment_method,
        recruitment_pool=args.recruitment_pool,
        n_max=args.n_max
    )
    
    # Print overall summary
    print("\nOverall Results:")
    for config_name, summary in results.get("summaries", {}).items():
        print(f"\n{config_name}:")
        for method, stats in summary.items():
            if "accuracy" in stats:
                print(f"  {method.replace('_', ' ').title()}: {stats['accuracy']:.2%} accuracy")


    # Print complexity metrics just once
    if "complexity_metrics" in results:
        print("\nComplexity Distribution:")
        for level in ["basic", "intermediate", "advanced"]:
            count = results["complexity_metrics"]["counts"].get(level, 0)
            correct = results["complexity_metrics"]["correct"].get(level, 0)
            accuracy = correct / count if count > 0 else 0.0
            print(f"  {level.title()}: {correct}/{count} correct ({accuracy:.2%})")

if __name__ == "__main__":
    main()


# Run 50 random MedQA questions with all teamwork components
#python dataset_runner.py --dataset medqa --num-questions 50 --leadership --closedloop --mutual --mental

# Run all configurations (baseline, individual components, all components) on PubMedQA
#python dataset_runner.py --dataset pubmedqa --num-questions 25 --all

# Specify custom output directory and random seed
#python dataset_runner.py --dataset medqa --output-dir ./results --seed 123 --all

# python dataset_runner.py --dataset medqa --output-dir ./medqa_results --seed 123 --num-questions 50 --leadership --closedloop --mutual --mental --all

# python dataset_runner.py --dataset medqa --output-dir ./medqa_results_rq --seed 123 --num-questions 5 --leadership --closedloop --mutual --mental --all --recruitment --recruitment-method adaptive --recruitment-pool medical


"""

Seed- 100, 200, 333

Single Agent ~ 10 mins run time
basic 1: python dataset_runner.py --dataset medqa --output-dir ./results_basic1 --seed 100 --num-questions 25 --recruitment --recruitment-method basic
basic 2: python dataset_runner.py --dataset medqa --output-dir ./results_basic2 --seed 200 --num-questions 25 --recruitment --recruitment-method basic
basic 3: python dataset_runner.py --dataset medqa --output-dir ./results_basic3 --seed 333 --num-questions 25 --recruitment --recruitment-method basic


Multi Agent ~ 30 mins run time
baseline 1: python dataset_runner.py --dataset medqa --output-dir ./results_baseline1 --seed 100 --num-questions 25 --recruitment
baseline 2: python dataset_runner.py --dataset medqa --output-dir ./results_baseline2 --seed 200 --num-questions 25 --recruitment
baseline 3: python dataset_runner.py --dataset medqa --output-dir ./results_baseline3 --seed 333 --num-questions 25 --recruitment


ABLATION STUDY ~ 40-50 mins run time
leadership 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_leadership --seed 100 --num-questions 25 --recruitment --leadership 
leadership 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_leadership2 --seed 200 --num-questions 25 --recruitment --leadership
leadership 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_leadership3 --seed 333 --num-questions 25 --recruitment --leadership

Trust 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_trust1 --seed 100 --num-questions 25 --recruitment --trust
Trust 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_trust2 --seed 200 --num-questions 25 --recruitment --trust
Trust 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_trust3 --seed 333 --num-questions 25 --recruitment --trust

Closed-Loop 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_loop1 --seed 100 --num-questions 25 --recruitment --closedloop
Closed-Loop 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_loop2 --seed 200 --num-questions 25 --recruitment --closedloop
Closed-Loop 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_loop3 --seed 333 --num-questions 25 --recruitment --closedloop

Mutual-Montioring 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_moniter1 --seed 100 --num-questions 25 --recruitment --mutual
Mutual-Montioring 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_moniter2 --seed 200 --num-questions 25 --recruitment --mutual
Mutual-Montioring 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_moniter3 --seed 333 --num-questions 25 --recruitment --mutual

mental model 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_mental1 --seed 100 --num-questions 25 --recruitment --mental
mental model 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_mental2 --seed 200 --num-questions 25 --recruitment --mental
mental model 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_mental3 --seed 333 --num-questions 25 --recruitment --mental

Orientation 1: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_orientation1 --seed 100 --num-questions 25 --recruitment --orientation
Orientation 2: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_orientation2 --seed 200 --num-questions 25 --recruitment --orientation
Orientation 3: python dataset_runner.py --dataset medqa --output-dir ./results_ablation_orientation3 --seed 333 --num-questions 25 --recruitment --orientation



COMBINED ~ 80 mins run time
all 1: python dataset_runner.py --dataset medqa --output-dir ./results_combined1 --seed 100 --num-questions 25 --recruitment --leadership --closedloop --mutual --mental --orientation --trust
all 2: python dataset_runner.py --dataset medqa --output-dir ./results_combined2 --seed 200 --num-questions 25 --recruitment --leadership --closedloop --mutual --mental --orientation --trust
all 3: python dataset_runner.py --dataset medqa --output-dir ./results_combined3 --seed 333 --num-questions 25 --recruitment --leadership --closedloop --mutual --mental --orientation --trust

"""