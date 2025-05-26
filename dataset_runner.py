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
from components.agent_recruitment import determine_complexity, recruit_agents

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

def load_medmcqa_dataset(num_questions: int = 50, random_seed: int = 42, include_multi_choice: bool = True) -> List[Dict[str, Any]]:
    """
    Load questions from the MedMCQA dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        include_multi_choice: Whether to include multi-choice questions
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MedMCQA dataset with {num_questions} random questions")
    
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        
        # Convert to list for easier processing
        questions = list(ds["train"])
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Filter questions based on choice type
        if include_multi_choice:
            # Include both single and multi-choice questions
            filtered_questions = questions
            logging.info(f"Including both single-choice and multi-choice questions")
        else:
            # Filter for single-choice questions only
            filtered_questions = [q for q in questions if q.get("choice_type") == "single"]
            logging.info(f"Filtering for single-choice questions only")
        
        if not filtered_questions:
            logging.error("No questions found after filtering")
            return []
        
        # Randomly select questions
        if num_questions < len(filtered_questions):
            selected_questions = random.sample(filtered_questions, num_questions)
        else:
            selected_questions = filtered_questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(filtered_questions)} questions. Using all available.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedMCQA dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedMCQA dataset: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def load_mmlupro_med_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load medical questions from the MMLU-Pro dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MMLU-Pro-Med dataset with {num_questions} random medical questions")
    
    try:
        # Try multiple approaches to load medical questions
        medical_questions = []
        
        # Approach 1: Try the specific medical dataset first
        try:
            ds = load_dataset("TIGER-Lab/MMLU-Pro")
            logging.info(f"Loaded TIGER-Lab/MMLU-Pro dataset")
            
            # Get all available splits
            available_splits = list(ds.keys())
            logging.info(f"Available splits: {available_splits}")
            
            # Medical-related categories in MMLU-Pro - comprehensive list
            medical_categories = [
                "anatomy", "clinical_knowledge", "medical_genetics", 
                "professional_medicine", "college_medicine", "college_biology",
                "high_school_biology", "nutrition", "virology", "health",
                "medicine", "biology", "psychology", "neuroscience",
                # Add more general categories that might contain medical questions
                "biochemistry", "pharmacology", "physiology", "pathology",
                "immunology", "microbiology", "epidemiology"
            ]
            
            # Filter for medical questions from all splits
            for split in available_splits:
                logging.info(f"Processing split: {split} with {len(ds[split])} questions")
                
                for item in ds[split]:
                    category = item.get("category", "").lower()
                    
                    # Check if any medical keyword is in the category
                    if any(cat in category for cat in medical_categories):
                        medical_questions.append(item)
                        if len(medical_questions) % 100 == 0:  # Log progress
                            logging.info(f"Found {len(medical_questions)} medical questions so far...")
                    
                    # Also check question content for medical terms if category doesn't match
                    elif category and len(medical_questions) < num_questions * 3:  # Don't over-process
                        question_text = item.get("question", "").lower()
                        medical_terms = [
                            "patient", "diagnosis", "treatment", "symptom", "disease",
                            "medication", "therapy", "clinical", "medical", "hospital",
                            "doctor", "physician", "nurse", "surgery", "cancer",
                            "infection", "virus", "bacteria", "immune", "blood",
                            "heart", "lung", "brain", "liver", "kidney"
                        ]
                        
                        if any(term in question_text for term in medical_terms):
                            medical_questions.append(item)
            
            logging.info(f"Found {len(medical_questions)} medical questions total")
            
        except Exception as e:
            logging.error(f"Failed to load TIGER-Lab dataset: {str(e)}")
            
            # Approach 2: Try alternative medical datasets
            try:
                logging.info("Trying alternative medical dataset...")
                ds = load_dataset("cais/mmlu", "anatomy")
                anatomy_questions = list(ds["test"]) + list(ds.get("validation", []))
                
                ds2 = load_dataset("cais/mmlu", "clinical_knowledge") 
                clinical_questions = list(ds2["test"]) + list(ds2.get("validation", []))
                
                ds3 = load_dataset("cais/mmlu", "medical_genetics")
                genetics_questions = list(ds3["test"]) + list(ds3.get("validation", []))
                
                ds4 = load_dataset("cais/mmlu", "professional_medicine")
                medicine_questions = list(ds4["test"]) + list(ds4.get("validation", []))
                
                medical_questions = anatomy_questions + clinical_questions + genetics_questions + medicine_questions
                logging.info(f"Loaded {len(medical_questions)} questions from MMLU medical subjects")
                
            except Exception as e2:
                logging.error(f"Failed to load alternative datasets: {str(e2)}")
                return []
        
        if not medical_questions:
            logging.error("No medical questions found in any MMLU dataset")
            return []
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(medical_questions):
            selected_questions = random.sample(medical_questions, num_questions)
        else:
            selected_questions = medical_questions
            logging.warning(f"Requested {num_questions} questions but found only {len(medical_questions)} medical questions. Using all available.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} medical questions from MMLU-Pro dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MMLU-Pro-Med dataset: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
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

def format_medmcqa_for_task(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format MedMCQA question for the agent system task.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Task dictionary for config.TASK
    """
    # Extract question text
    question_text = question_data.get("question", "")
    
    # Extract options
    opa = question_data.get("opa", "")
    opb = question_data.get("opb", "")
    opc = question_data.get("opc", "")
    opd = question_data.get("opd", "")
    
    # Format as standard MCQ options
    options = [
        f"A. {opa}",
        f"B. {opb}",
        f"C. {opc}",
        f"D. {opd}"
    ]
    
    # Check if it's a multi-choice question
    choice_type = question_data.get("choice_type", "single")
    
    if choice_type == "single":
        # Get correct option (cop is 1-indexed, so cop=1 means option A)
        cop = question_data.get("cop", 0)
        ground_truth = chr(64 + cop) if 1 <= cop <= 4 else "A"  # Convert 1->A, 2->B, etc.
        task_type = "mcq"
        expected_format = "Single letter selection with rationale"
    else:
        # Multi-choice question - parse the correct answers
        cop = question_data.get("cop", "")
        
        # Handle different formats for multi-choice answers
        if isinstance(cop, int):
            # If it's still a single int for multi-choice, convert it
            ground_truth = chr(64 + cop) if 1 <= cop <= 4 else "A"
        elif isinstance(cop, str):
            # Parse string like "1,2" or "A,B" 
            if ',' in cop:
                parts = [p.strip() for p in cop.split(',')]
                ground_truth_list = []
                for part in parts:
                    if part.isdigit() and 1 <= int(part) <= 4:
                        ground_truth_list.append(chr(64 + int(part)))
                    elif part.upper() in ['A', 'B', 'C', 'D']:
                        ground_truth_list.append(part.upper())
                ground_truth = ','.join(sorted(ground_truth_list)) if ground_truth_list else cop
            else:
                # Single answer in string format
                if cop.isdigit() and 1 <= int(cop) <= 4:
                    ground_truth = chr(64 + int(cop))
                elif cop.upper() in ['A', 'B', 'C', 'D']:
                    ground_truth = cop.upper()
                else:
                    ground_truth = cop
        else:
            # Default to string representation
            ground_truth = str(cop)
            
        task_type = "multi_choice_mcq"
        expected_format = "Multiple letter selections (e.g., A,C or A,B,D) with rationale"
        
        # Add note about multi-choice in description
        question_text = f"[MULTI-CHOICE QUESTION - Select ALL correct options]\n\n{question_text}"
    
    # Get explanation and metadata
    explanation = question_data.get("exp", "")
    subject_name = question_data.get("subject_name", "")
    topic_name = question_data.get("topic_name", "")
    
    # Create task dictionary
    task = {
        "name": "MedMCQA Question",
        "description": question_text,
        "type": task_type,
        "options": options,
        "expected_output_format": expected_format,
        "ground_truth": ground_truth,
        "rationale": {ground_truth: explanation} if explanation else {},
        "metadata": {
            "subject": subject_name,
            "topic": topic_name,
            "question_id": question_data.get("id", ""),
            "choice_type": choice_type
        }
    }
    
    return task

def format_mmlupro_med_for_task(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format MMLU-Pro medical question for the agent system task.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Task dictionary for config.TASK
    """
    # Extract question text
    question_text = question_data.get("question", "")
    
    # Extract options - handle both dict and list formats
    options_data = question_data.get("options", {})
    options = []
    
    if isinstance(options_data, dict):
        # Format options as standard MCQ options from dict
        for key in sorted(options_data.keys()):
            options.append(f"{key}. {options_data[key]}")
    elif isinstance(options_data, list):
        # Format options as standard MCQ options from list
        for i, option in enumerate(options_data):
            options.append(f"{chr(65+i)}. {option}")
    
    # Get correct answer
    ground_truth = question_data.get("answer", "")
    
    # Handle different answer formats
    if not ground_truth and "answer_idx" in question_data:
        answer_idx = question_data.get("answer_idx")
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
            ground_truth = chr(65 + answer_idx)  # Convert 0->A, 1->B, etc.
    
    # Get answer index if available
    answer_idx = question_data.get("answer_idx", "")
    
    # Create task dictionary
    task = {
        "name": "MMLU-Pro Medical Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale",
        "ground_truth": ground_truth,
        "rationale": {},  # MMLU-Pro doesn't provide explanations
        "metadata": {
            "category": question_data.get("category", ""),
            "answer_idx": answer_idx
        }
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
    elif isinstance(context, list):
        # Some versions of the dataset have context as a list
        context_text = "\n\n".join(context)
    elif isinstance(context, str):
        context_text = context
    
    # Get the expected output (correct answer)
    expected_output = question_data.get("final_decision", "").lower()
    
    # Create task dictionary
    task = {
        "name": "PubMedQA Question",
        "description": f"Research Question: {question_text}\n\nAbstract Context:\n{context_text}",
        "type": "yes_no_maybe",
        "options": ["yes", "no", "maybe"],
        "expected_output_format": "Answer (yes/no/maybe) with detailed scientific justification",
        "ground_truth": expected_output,
        "rationale": {"long_answer": question_data.get("long_answer", "")},
        "metadata": {
            "pubid": question_data.get("pubid", "")
        }
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

    # Make sure configuration has proper recruitment method and n_max
    if config_name == "Baseline":
        # Baseline always uses basic recruitment with 1 agent
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "basic"
        configuration["n_max"] = 1
    elif config_name != "Custom Configuration":
        # All other named configurations use intermediate recruitment with specified n_max
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "intermediate"
        if "n_max" not in configuration:
            configuration["n_max"] = n_max
    
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
    
    # Add summary for yes/no/maybe if PubMedQA
    if dataset_type == "pubmedqa":
        results["summary"]["yes_no_maybe_voting"] = {"correct": 0, "total": 0}
    
    # Process each question
    for i, question in enumerate(tqdm(questions, desc=f"{config_name}")):
        question_result = {"question_index": i}
        simulator = None  # Initialize simulator variable
        performance = None  # Initialize performance variable
        
        try:
            # Format the question for the task
            if dataset_type == "medqa":
                task = format_medqa_for_task(question)
            elif dataset_type == "medmcqa":
                task = format_medmcqa_for_task(question)
            elif dataset_type == "pubmedqa":
                task = format_pubmedqa_for_task(question)
            elif dataset_type == "mmlupro-med":
                task = format_mmlupro_med_for_task(question)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            question_result["question"] = task["description"]
            question_result["ground_truth"] = task.get("ground_truth", "")
            
            # Update task configuration
            config.TASK = task
            
            # Try to run the simulation with retries
            for attempt in range(max_retries):
                try:
                    # Create simulator with configuration parameters
                    simulator = AgentSystemSimulator(
                        simulation_id=f"{dataset_type}_{config_name.lower().replace(' ', '_')}_{i}",
                        use_team_leadership=configuration.get("leadership", False),
                        use_closed_loop_comm=configuration.get("closed_loop", False),
                        use_mutual_monitoring=configuration.get("mutual_monitoring", False),
                        use_shared_mental_model=configuration.get("shared_mental_model", False),
                        use_team_orientation=configuration.get("team_orientation", False),
                        use_mutual_trust=configuration.get("mutual_trust", False),
                        use_recruitment=configuration.get("recruitment", False),
                        recruitment_method=configuration.get("recruitment_method", "adaptive"),
                        recruitment_pool=configuration.get("recruitment_pool", "general"),
                        n_max=configuration.get("n_max", n_max)  # Use configuration-specific n_max or default
                    )
                    
                    # Run simulation
                    simulation_results = simulator.run_simulation()
                    performance = simulator.evaluate_performance()
                    
                    # Store decisions and performance
                    question_result["decisions"] = simulation_results["decision_results"]
                    question_result["performance"] = performance.get("task_performance", {})
                    
                    # Store all agent conversations
                    question_result["agent_conversations"] = simulation_results.get("exchanges", [])
                    
                    if performance is not None:
                        # Update summary statistics
                        task_performance = performance.get("task_performance", {})
                        
                        # Handle different task types properly
                        if dataset_type == "pubmedqa":
                            # For PubMedQA, look for yes_no_maybe_voting method
                            for method in ["majority_voting", "weighted_voting", "borda_count"]:
                                if method in task_performance:
                                    method_perf = task_performance[method]
                                    if "correct" in method_perf:
                                        results["summary"][method]["total"] += 1
                                        if method_perf["correct"]:
                                            results["summary"][method]["correct"] += 1
                            
                            # Special handling for yes_no_maybe_voting if it exists
                            if "yes_no_maybe_voting" in task_performance:
                                yes_no_perf = task_performance["yes_no_maybe_voting"]
                                if "correct" in yes_no_perf:
                                    results["summary"]["yes_no_maybe_voting"]["total"] += 1
                                    if yes_no_perf["correct"]:
                                        results["summary"]["yes_no_maybe_voting"]["correct"] += 1
                        else:
                            # For other datasets, handle normally
                            methods_to_check = ["majority_voting", "weighted_voting", "borda_count"]
                                
                            for method in methods_to_check:
                                if method in task_performance:
                                    method_perf = task_performance[method]
                                    if "correct" in method_perf:
                                        results["summary"][method]["total"] += 1
                                        if method_perf["correct"]:
                                            results["summary"][method]["correct"] += 1
                    
                    # Simulation succeeded, break the retry loop
                    break
                
                except Exception as e:
                    error_str = str(e)
                    import traceback
                    error_details = traceback.format_exc()
                    logging.error(f"Full error details: {error_details}")
                    
                    # Check for different error types
                    if attempt < max_retries - 1:
                        # Content filter errors
                        if "content" in error_str.lower() and "filter" in error_str.lower():
                            error_type = "content_filter"
                            wait_time = min(5 * (attempt + 1), 30)
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
                            "ground_truth": task.get("ground_truth", ""),
                            "metadata": task.get("metadata", {})
                        }
                        if "decisions" in question_result:
                            output_data["decisions"] = question_result["decisions"]
                        if "performance" in question_result:
                            output_data["performance"] = question_result["performance"]
                        if "agent_conversations" in question_result:
                            output_data["agent_conversations"] = question_result["agent_conversations"]
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
    for method in results["summary"].keys():
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
                
            # Save detailed results with conversations
            with open(os.path.join(run_output_dir, "detailed_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save summary results: {str(e)}")
    
    # Print summary
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")
    
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")

    # Add complexity metrics before returning results
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
    n_max: int = 5
) -> Dict[str, Any]:
    """
    Run a dataset through the agent system.
    
    Args:
        dataset_type: Type of dataset ("medqa", "medmcqa", "pubmedqa", "mmlupro-med")
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
        n_max: Maximum number of agents for intermediate teams
        
    Returns:
        Results dictionary
    """
    # Ensure n_max has a default value
    if n_max is None:
        n_max = 5
    
    # Log parameters
    logging.info(f"Running dataset: {dataset_type}, n_max={n_max}, recruitment_method={recruitment_method}")

    # Load the dataset
    if dataset_type == "medqa":
        questions = load_medqa_dataset(num_questions, random_seed)
    elif dataset_type == "medmcqa":
        questions = load_medmcqa_dataset(num_questions, random_seed, include_multi_choice=True)
    elif dataset_type == "pubmedqa":
        questions = load_pubmedqa_dataset(num_questions, random_seed)
    elif dataset_type == "mmlupro-med":
        questions = load_mmlupro_med_dataset(num_questions, random_seed)
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
            # Baseline - a single Medical Generalist agent (using basic recruitment)
            {
                "name": "Baseline", 
                "description": "Single Medical Generalist agent, no teamwork components",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,  # Set to True but will be handled specially
                "recruitment_method": "basic",  # Force basic recruitment for Baseline
                "n_max": 1  # Always use just 1 agent for Baseline
            },

            # Standard Team - uses specified n_max agents (using intermediate recruitment)
            {
                "name": "Standard Team", 
                "description": f"Team of {n_max} agents with no teamwork components, using intermediate recruitment",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",  # Use intermediate recruitment method
                "recruitment_pool": recruitment_pool,
                "n_max": n_max  # Use specified n_max value
            },

            # Single features with intermediate recruitment
            {
                "name": "Leadership", 
                "description": f"Team of {n_max} agents with leadership component",
                "leadership": True, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            {
                "name": "Closed-loop", 
                "description": f"Team of {n_max} agents with closed-loop communication",
                "leadership": False, 
                "closed_loop": True,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            {
                "name": "Mutual Monitoring", 
                "description": f"Team of {n_max} agents with mutual monitoring",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": True,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            {
                "name": "Shared Mental Model", 
                "description": f"Team of {n_max} agents with shared mental model",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": True,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            {
                "name": "Team Orientation", 
                "description": f"Team of {n_max} agents with team orientation",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": True,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            {
                "name": "Mutual Trust", 
                "description": f"Team of {n_max} agents with mutual trust",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max
            },
            # All features with recruitment
            {
                "name": "All Features with Recruitment", 
                "description": f"Team of {n_max} agents with all teamwork components",
                "leadership": True, 
                "closed_loop": True,
                "mutual_monitoring": True,
                "shared_mental_model": True,
                "team_orientation": True,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
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
            "recruitment_pool": recruitment_pool,
            "n_max": n_max
        }]
    
    # Run each configuration
    all_results = []
    for config in configurations:
        # Ensure each config has proper n_max value
        if "n_max" not in config:
            config["n_max"] = n_max
        
        # Special handling for Baseline - always use basic recruitment with 1 agent
        if config["name"] == "Baseline":
            config["recruitment"] = True
            config["recruitment_method"] = "basic"
            config["n_max"] = 1
        
        # Add recruitment settings to all configs if recruitment is enabled
        if config["recruitment"]:
            config["recruitment_method"] = config.get("recruitment_method", recruitment_method)
            config["recruitment_pool"] = config.get("recruitment_pool", recruitment_pool)
        
        # Log current configuration with description if available
        description = config.get("description", "")
        desc_str = f" - {description}" if description else ""
        logging.info(f"Running configuration: {config['name']}{desc_str}, recruitment={config['recruitment']}, method={config['recruitment_method']}, n_max={config['n_max']}")
        
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config,
            run_output_dir,
            n_max=config.get("n_max", n_max)
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
    parser.add_argument('--dataset', type=str, default="medqa", 
                      choices=["medqa", "medmcqa", "pubmedqa", "mmlupro-med"], 
                      help='Dataset to run (medqa, medmcqa, pubmedqa, or mmlupro-med)')
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
                      default='medical' if '--dataset' in sys.argv and sys.argv[sys.argv.index('--dataset')+1] in ['medqa', 'medmcqa', 'pubmedqa'] else 'general', 
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


# Example usage:
# Run 50 random MedMCQA questions with all teamwork components
# python dataset_runner.py --dataset medmcqa --num-questions 50 --leadership --closedloop --mutual --mental

# Run all configurations on PubMedQA
# python dataset_runner.py --dataset pubmedqa --num-questions 25 --all

# Run with recruitment on MedMCQA
# python dataset_runner.py --dataset medmcqa --output-dir ./medmcqa_results --seed 123 --recruitment --recruitment-method adaptive --recruitment-pool medical