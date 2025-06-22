"""
Dataset runner for the modular agent system with fixed question-level parallel processing.
Runs multiple questions from datasets through the agent system with proper isolation.
"""

import argparse
import logging
import os
import json
import random
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from tqdm import tqdm
import concurrent.futures
import threading
import copy

from datasets import load_dataset
from simulator import AgentSystemSimulator
import config
from utils.logger import SimulationLogger
from components.agent_recruitment import determine_complexity, recruit_agents
from components.medrag_integration import create_medrag_integration

# Thread-local storage for progress tracking
thread_local = threading.local()

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
    
    Note: The include_multi_choice parameter is kept for compatibility but all questions
    are actually single-choice due to the dataset structure.
    """
    logging.info(f"Loading MedMCQA dataset with {num_questions} random questions")
    
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        questions = list(ds["train"])
        
        # Log the choice type distribution for awareness
        choice_types = [q.get("choice_type", "unknown") for q in questions[:1000]]  # Sample first 1000
        choice_type_counts = {}
        for ct in choice_types:
            choice_type_counts[ct] = choice_type_counts.get(ct, 0) + 1
        
        logging.info(f"Choice type distribution in dataset sample: {choice_type_counts}")
        logging.info("Note: All questions will be treated as single-choice due to known dataset issue")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions (no need to filter by choice_type)
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedMCQA dataset")
        validate_medmcqa_parsing(selected_questions)
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedMCQA dataset: {str(e)}")
        return []

def validate_medmcqa_parsing(sample_questions: List[Dict[str, Any]]) -> None:
    """Validate that MedMCQA parsing works correctly for all question types."""
    
    print("Validating MedMCQA parsing...")
    
    single_count = 0
    multi_count = 0
    errors = 0
    
    for i, question_data in enumerate(sample_questions[:20]):  # Test first 20
        try:
            original_type = question_data.get("choice_type", "unknown")
            original_cop = question_data.get("cop", "N/A")
            
            # Parse with new function
            task, eval_data = format_medmcqa_for_task(question_data)
            parsed_answer = eval_data.get("ground_truth", "ERROR")
            
            if original_type == "single":
                single_count += 1
            elif original_type == "multi":
                multi_count += 1
            
            print(f"Q{i+1}: {original_type} -> {parsed_answer} (cop: {original_cop})")
            
        except Exception as e:
            errors += 1
            print(f"Q{i+1}: ERROR - {e}")
    
    print(f"\nSummary: {single_count} single, {multi_count} multi, {errors} errors")
    print("All questions are now correctly parsed as single-choice MCQs.")

def load_mmlupro_med_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Health questions from the MMLU-Pro dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MMLU-Pro Health dataset with {num_questions} random questions")
    
    try:
        # Load the MMLU-Pro dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        logging.info(f"Loaded TIGER-Lab/MMLU-Pro dataset")
        
        # Get all available splits
        available_splits = list(ds.keys())
        logging.info(f"Available splits: {available_splits}")
        
        # Filter for Health category only
        health_questions = []
        
        for split in available_splits:
            logging.info(f"Processing split: {split} with {len(ds[split])} questions")
            
            for item in ds[split]:
                category = item.get("category", "").lower()
                
                # Filter specifically for health category
                if "health" in category:
                    health_questions.append(item)
                    if len(health_questions) % 50 == 0:  # Log progress
                        logging.info(f"Found {len(health_questions)} health questions so far...")
        
        logging.info(f"Found {len(health_questions)} health questions total")
        
        if not health_questions:
            logging.error("No health questions found in MMLU-Pro dataset")
            return []
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(health_questions):
            selected_questions = random.sample(health_questions, num_questions)
        else:
            selected_questions = health_questions
            logging.warning(f"Requested {num_questions} questions but found only {len(health_questions)} health questions. Using all available.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} health questions from MMLU-Pro dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MMLU-Pro Health dataset: {str(e)}")
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

def format_medqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MedQA question into agent task and evaluation data.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input (no GT)
        - eval_data: Ground truth, rationale, metadata for evaluation
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
    
    # Create agent task (no ground truth)
    agent_task = {
        "name": "MedQA Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale"
    }
    
    # Create evaluation data (with ground truth)
    eval_data = {
        "ground_truth": expected_output,
        "rationale": {},  # No rationale provided in the dataset
        "metadata": {
            "dataset": "MedQA",
            "question_id": question_data.get("id", ""),
            "metamap": question_data.get("metamap", ""),
            "answer_idx": question_data.get("answer_idx", "")
        }
    }
    
    return agent_task, eval_data

def format_medmcqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MedMCQA question into agent task and evaluation data.
    Enhanced to provide better context for agent recruitment.

    Args:
        question_data: Question data from the MedMCQA dataset.

    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input.
        - eval_data: Ground truth, rationale, metadata for evaluation.
    """
    question_text = question_data.get("question", "")
    opa, opb, opc, opd = question_data.get("opa", ""), question_data.get("opb", ""), question_data.get("opc", ""), question_data.get("opd", "")
    option_letters = ['A', 'B', 'C', 'D']
    option_values = [opa, opb, opc, opd]

    options = [f"{letter}. {value}" for letter, value in zip(option_letters, option_values) if value.strip()]
    original_choice_type = question_data.get("choice_type", "single")
    
    # Enhanced description for recruitment context
    enhanced_description = question_text
    
    # If question is very short/vague, add options context for better recruitment
    if len(question_text.split()) < 10:  # Short questions likely need context
        options_text = " | ".join([f"{letter}: {value}" for letter, value in zip(option_letters, option_values) if value.strip()])
        enhanced_description = f"{question_text}\n\nAnswer options provide context: {options_text}"
    
    if original_choice_type == "multi":
        logging.debug(f"Question ID {question_data.get('id', 'unknown')} labeled as 'multi', treating as single choice.")
    
    cop = question_data.get("cop", 1)
    ground_truth = parse_cop_field(cop)

    explanation = question_data.get("exp", "")
    subject_name = question_data.get("subject_name", "")
    topic_name = question_data.get("topic_name", "")

    agent_task = {
        "name": "MedMCQA Question",
        "description": enhanced_description,  # Use enhanced description
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale",
        "subject_context": subject_name,  # Add for recruitment
        "topic_context": topic_name       # Add for recruitment
    }

    eval_data = {
        "ground_truth": ground_truth,
        "rationale": {ground_truth: explanation} if explanation else {},
        "metadata": {
            "subject": subject_name,
            "topic": topic_name,
            "question_id": question_data.get("id", ""),
            "original_choice_type": original_choice_type,
            "cop_original": question_data.get("cop", ""),
            "original_question": question_text,  # Keep original for reference
            "enhanced_for_recruitment": len(question_text.split()) < 10
        }
    }

    return agent_task, eval_data


def parse_cop_field(cop) -> str:
    """
    Robustly parse the cop (correct option) field from MedMCQA dataset.
    
    Args:
        cop: The cop field value (can be int, str, or other)
        
    Returns:
        Single letter representing the correct answer (A, B, C, or D)
    """
    # Handle integer values (most common case)
    if isinstance(cop, int):
        return chr(64 + cop) if 1 <= cop <= 4 else "A"
    
    # Handle string values
    elif isinstance(cop, str):
        cop = cop.strip()
        
        # Handle numeric strings
        if cop.isdigit():
            cop_int = int(cop)
            return chr(64 + cop_int) if 1 <= cop_int <= 4 else "A"
        
        # Handle letter answers
        elif cop.upper() in ['A', 'B', 'C', 'D']:
            return cop.upper()
        
        # Handle comma-separated (take first one - fallback for edge cases)
        elif ',' in cop:
            first_part = cop.split(',')[0].strip()
            return parse_cop_field(first_part)  # Recursive call
        
        # Handle empty strings
        elif not cop:
            return "A"
        
        # Other string formats - try to extract number
        else:
            import re
            match = re.search(r'\d+', cop)
            if match:
                cop_int = int(match.group())
                return chr(64 + cop_int) if 1 <= cop_int <= 4 else "A"
            else:
                return "A"  # Default fallback
    
    # Handle list/array values (edge case)
    elif isinstance(cop, (list, tuple)) and len(cop) > 0:
        return parse_cop_field(cop[0])  # Take first element
    
    # Handle other types
    else:
        # Try to convert to string and parse
        try:
            return parse_cop_field(str(cop))
        except:
            return "A"  # Final fallback


def format_mmlupro_med_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MMLU-Pro medical question into agent task and evaluation data.
    Supports up to 10 options (A-J).

    Args:
        question_data: Question data from the MMLU-Pro dataset.

    Returns:
        Tuple of:
        - agent_task: Task for the agent system.
        - eval_data: Ground truth, rationale, and metadata.
    """
    question_text = question_data.get("question", "")
    options_data = question_data.get("options", [])
    options = []

    # Handle list format with up to 10 options
    if isinstance(options_data, list):
        for i, option in enumerate(options_data):
            if i < 10:  # Support up to 10 options (A-J)
                letter = chr(65 + i)  # A=65, B=66, ..., J=74
                options.append(f"{letter}. {option}")
            else:
                break  # Don't exceed 10 options
    elif isinstance(options_data, dict):
        # Handle dict format
        for key in sorted(options_data.keys()):
            options.append(f"{key}. {options_data[key]}")

    # Get ground truth
    ground_truth = question_data.get("answer", "")
    if not ground_truth and "answer_idx" in question_data:
        idx = question_data["answer_idx"]
        if isinstance(idx, int) and 0 <= idx < len(options_data):
            ground_truth = chr(65 + idx)  # Convert index to letter

    # Create expected output format based on number of options
    num_options = len(options)
    if num_options <= 4:
        expected_format = "Single letter selection (A-D) with rationale"
    elif num_options <= 10:
        last_letter = chr(64 + num_options)  # 64 + 10 = J
        expected_format = f"Single letter selection (A-{last_letter}) with rationale"
    else:
        expected_format = "Single letter selection with rationale"

    agent_task = {
        "name": "MMLU-Pro Health Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": expected_format,
        "num_options": num_options  # Add this for agents to know
    }

    eval_data = {
        "ground_truth": ground_truth,
        "rationale": {},  # MMLU-Pro provides no rationale
        "metadata": {
            "category": question_data.get("category", ""),
            "answer_idx": question_data.get("answer_idx", ""),
            "num_options": num_options
        }
    }

    return agent_task, eval_data

def format_pubmedqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format PubMedQA question into agent task and evaluation data.

    Args:
        question_data: Question data from PubMedQA.

    Returns:
        Tuple of:
        - agent_task: Task input for the agent.
        - eval_data: Ground truth and explanation for evaluation.
    """
    question_text = question_data.get("question", "")
    context = question_data.get("context", "")
    context_text = ""

    if isinstance(context, dict):
        for section, text in context.items():
            context_text += f"{section}: {text}\n\n"
    elif isinstance(context, list):
        context_text = "\n\n".join(context)
    elif isinstance(context, str):
        context_text = context

    expected_output = question_data.get("final_decision", "").lower()

    agent_task = {
        "name": "PubMedQA Question",
        "description": f"Research Question: {question_text}\n\nAbstract Context:\n{context_text}",
        "type": "yes_no_maybe",
        "options": ["yes", "no", "maybe"],
        "expected_output_format": "Answer (yes/no/maybe) with detailed scientific justification"
    }

    eval_data = {
        "ground_truth": expected_output,
        "rationale": {"long_answer": question_data.get("long_answer", "")},
        "metadata": {
            "pubid": question_data.get("pubid", "")
        }
    }

    return agent_task, eval_data

def detect_agent_disagreement(agent_responses):
    """Detect if agents disagree on answers"""
    answers = []
    for agent_role, response in agent_responses.items():
        answer = None
        if isinstance(response, dict):
            if "final_decision" in response:
                # Extract answer from final_decision text
                answer = extract_answer_option(response["final_decision"])
            elif "extract" in response and isinstance(response["extract"], dict):
                if "answer" in response["extract"]:
                    answer = response["extract"]["answer"]
        elif isinstance(response, str):
            answer = extract_answer_option(response)
        
        if answer:
            answers.append(answer.upper())
    
    unique_answers = set(answers)
    disagreement = len(unique_answers) > 1
    
    return {
        "has_disagreement": disagreement,
        "unique_answers": list(unique_answers),
        "answer_distribution": {ans: answers.count(ans) for ans in unique_answers},
        "total_agents": len(answers),
        "all_answers": answers
    }

def extract_answer_option(content):
    """Extract answer option from agent response content"""
    if not isinstance(content, str):
        return None
    
    import re
    patterns = [
        r"ANSWER:\s*([A-Da-d])",
        r"FINAL ANSWER:\s*([A-Da-d])",
        r"answer is:?\s*([A-Da-d])",
        r"my answer:?\s*([A-Da-d])",
        r"option\s+([A-Da-d])",
        r"choose\s+([A-Da-d])",
        r"select\s+([A-Da-d])",
        r"\*\*([A-Da-d])\.\*\*",
        r"^([A-Da-d])\.",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def extract_yes_no_maybe_from_text(content):
    """Extract yes/no/maybe from text content."""
    if not isinstance(content, str):
        return None
    
    import re
    patterns = [
        r"ANSWER:\s*(yes|no|maybe)",
        r"answer is:?\s*(yes|no|maybe)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content.lower())
        if match:
            return match.group(1).lower()
    return None


def extract_agent_responses_info(simulation_results):
    """Extract detailed agent response information"""
    agent_responses_info = {}
    agent_responses = simulation_results.get("agent_responses", {})
    
    for agent_role, response_data in agent_responses.items():
        info = {
            "final_answer": None,
            "confidence": 1.0,
            "reasoning": "",
            "full_response": response_data
        }
        
        if isinstance(response_data, dict):
            # Extract answer
            if "final_decision" in response_data:
                # Try MCQ first, then yes/no/maybe
                info["final_answer"] = extract_answer_option(response_data["final_decision"])
                if not info["final_answer"]:
                    info["final_answer"] = extract_yes_no_maybe_from_text(response_data["final_decision"])
                info["reasoning"] = response_data["final_decision"][:500] + "..." if len(response_data["final_decision"]) > 500 else response_data["final_decision"]
            elif "extract" in response_data and isinstance(response_data["extract"], dict):
                if "final_decision" in response_data:
                    info["final_answer"] = extract_answer_option(response_data["final_decision"])
                if "confidence" in response_data["extract"]:
                    info["confidence"] = response_data["extract"]["confidence"]
            
            # Extract confidence if available
            if "confidence" in response_data:
                info["confidence"] = response_data["confidence"]
        else:
            info["final_answer"] = extract_answer_option(str(response_data))
            info["reasoning"] = str(response_data)[:500] + "..." if len(str(response_data)) > 500 else str(response_data)
        
        agent_responses_info[agent_role] = info
    
    return agent_responses_info

def process_single_question(question_index: int, 
                            question: Dict[str, Any],
                            dataset_type: str,
                            configuration: Dict[str, Any],
                            deployment_config: Dict[str, str],
                            run_output_dir: str,
                            max_retries: int = 3,
                            use_medrag: bool = False) -> Dict[str, Any]:
    """
    Process a single question with the given configuration and deployment.
    Creates isolated simulation context for the question.

    Args:
        question_index: Index of the question in the dataset
        question: Question data
        dataset_type: Type of dataset (medqa, medmcqa, etc.)
        configuration: Configuration dictionary with teamwork settings
        deployment_config: Specific deployment configuration to use
        run_output_dir: Output directory for results
        max_retries: Maximum number of retries for failed questions

    Returns:
        Dictionary with question results
    """
    import gc
    import traceback

    # Create unique simulation ID for this question
    sim_id = f"{dataset_type}_{configuration['name'].lower().replace(' ', '_')}_q{question_index}_{deployment_config['name']}"
    if use_medrag:
        sim_id += "_medrag"
    
    # Format question based on dataset type
    if dataset_type == "medqa":
        agent_task, eval_data = format_medqa_for_task(question)
    elif dataset_type == "medmcqa":
        agent_task, eval_data = format_medmcqa_for_task(question)
    elif dataset_type == "mmlupro-med":
        agent_task, eval_data = format_mmlupro_med_for_task(question)
    elif dataset_type == "pubmedqa":
        agent_task, eval_data = format_pubmedqa_for_task(question)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Set thread-local data for this question
    thread_local.question_index = question_index
    thread_local.question_task = agent_task
    thread_local.question_eval = eval_data

    question_result = {
        "question_index": question_index,
        "deployment_used": deployment_config['name'],
        "simulation_id": sim_id,
        "recruitment_info": {},
        "agent_responses": {},
        "disagreement_analysis": {},
        "disagreement_flag": False,
        "medrag_info": {}  # Store MedRAG information
    }

    try:
        for attempt in range(max_retries):
            try:
                # Create simulator with isolated task configuration
                simulator = AgentSystemSimulator(
                    simulation_id=sim_id,
                    use_team_leadership=configuration.get("leadership", False),
                    use_closed_loop_comm=configuration.get("closed_loop", False),
                    use_mutual_monitoring=configuration.get("mutual_monitoring", False),
                    use_shared_mental_model=configuration.get("shared_mental_model", False),
                    use_team_orientation=configuration.get("team_orientation", False),
                    use_mutual_trust=configuration.get("mutual_trust", False),
                    use_recruitment=configuration.get("recruitment", False),
                    recruitment_method=configuration.get("recruitment_method", "adaptive"),
                    recruitment_pool=configuration.get("recruitment_pool", "general"),
                    n_max=configuration.get("n_max", 5),
                    deployment_config=deployment_config,
                    question_specific_context=True,
                    task_config=agent_task,  # Pass task directly to simulator
                    eval_data=eval_data  ,    # Pass evaluation data directly
                    use_medrag=use_medrag  # Pass MedRAG parameter
                )

                # Log that we're starting processing for this question
                medrag_status = "with MedRAG" if use_medrag else "without MedRAG"
                logging.info(f"Processing question {question_index} {medrag_status}, deployment {deployment_config['name']}, sim_id: {sim_id}")

                simulation_results = simulator.run_simulation()
                performance = simulator.evaluate_performance()


                # Extract MedRAG information if used
                if use_medrag and "medrag_enhancement" in simulation_results.get("simulation_metadata", {}):
                    question_result["medrag_info"] = simulation_results["simulation_metadata"]["medrag_enhancement"]

                question_result.update({
                    "agent_responses": extract_agent_responses_info(simulation_results),
                    "disagreement_analysis": detect_agent_disagreement(simulation_results.get("agent_responses", {})),
                    "decisions": simulation_results.get("decision_results", {}),
                    "performance": performance.get("task_performance", {}),
                    "agent_conversations": simulation_results.get("exchanges", [])
                })

                if question_result["disagreement_analysis"].get("has_disagreement", False):
                    question_result["disagreement_flag"] = True

                # Save individual question result to file
                if run_output_dir:
                    question_result_file = os.path.join(run_output_dir, f"question_{question_index}_result.json")
                    with open(question_result_file, 'w') as f:
                        json.dump(question_result, f, indent=2)
                    logging.info(f"Saved question {question_index} result to {question_result_file}")

                break  # Success, break retry loop

            except Exception as e:
                logging.error(f"Question {question_index} attempt {attempt+1} failed: {str(e)}")
                logging.error(traceback.format_exc())
                time.sleep(min(2 ** attempt, 15))
        
    except Exception as e:
        logging.error(f"Final error processing question {question_index}: {str(e)}")
        question_result["error"] = f"Processing error: {str(e)}"

    finally:
        # Clean up thread-local data and force garbage collection
        if hasattr(thread_local, 'question_task'):
            delattr(thread_local, 'question_task')
        if hasattr(thread_local, 'question_eval'):
            delattr(thread_local, 'question_eval')
        
        if 'simulator' in locals():
            del simulator
        gc.collect()

    return question_result

def track_agent_mind_changes(agent_analyses, agent_decisions):
    """Track if agents changed their answers after seeing teammates"""
    mind_changes = {}
    
    for agent_role in agent_analyses.keys():
        # Extract initial answer
        initial_extract = agent_analyses[agent_role].get("extract", {})
        initial_answer = initial_extract.get("answer", "").upper() if initial_extract.get("answer") else ""
        
        # Extract final answer
        final_extract = agent_decisions[agent_role].get("extract", {})
        final_answer = final_extract.get("answer", "").upper() if final_extract.get("answer") else ""
        
        # Check if changed
        changed = initial_answer != final_answer and initial_answer and final_answer
        
        mind_changes[agent_role] = {
            "initial_answer": initial_answer,
            "final_answer": final_answer, 
            "changed_mind": changed,
            "change_direction": f"{initial_answer} → {final_answer}" if changed else None
        }
    
    return mind_changes

def run_questions_with_configuration(
    questions: List[Dict[str, Any]],
    dataset_type: str,
    configuration: Dict[str, bool],
    output_dir: Optional[str] = None,
    max_retries: int = 3,
    n_max: int = 5,
    use_medrag: bool = False  # NEW PARAMETER
) -> Dict[str, Any]:
    """
    Run questions with specific configuration using parallel processing at question level.
    Each question is assigned to a deployment in round-robin fashion.
    
    Args:
        questions: List of questions to process
        dataset_type: Type of dataset
        configuration: Configuration dictionary
        output_dir: Output directory
        max_retries: Maximum retries per question
        n_max: Maximum number of agents for intermediate teams
        
    Returns:
        Dictionary with aggregated results
    """
    config_name = configuration.get("name", "unknown")
    if use_medrag:
        config_name += "_medrag"
    
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
    # Get all available deployments
    deployments = config.get_all_deployments()
    num_deployments = len(deployments)
    
    logging.info(f"Using {num_deployments} deployments for parallel question processing: {[d['name'] for d in deployments]}")
    
    # Reset complexity metrics
    from components import agent_recruitment
    agent_recruitment.reset_complexity_metrics()

    # Make sure configuration has proper recruitment method and n_max
    if config_name == "Baseline":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "basic"
        configuration["n_max"] = 1
    elif config_name != "Custom Configuration":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "intermediate"
        if "n_max" not in configuration:
            configuration["n_max"] = n_max
    
    # Setup output directory
    run_output_dir = os.path.join(output_dir, f"{dataset_type}_{config_name.lower().replace(' ', '_')}") if output_dir else None
    if run_output_dir:
        os.makedirs(run_output_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        "configuration": config_name,
        "use_medrag": use_medrag,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "configuration_details": configuration,
        "deployment_info": {
            "deployments_used": deployments,
            "parallel_processing": "question_level",
            "num_parallel_questions": num_deployments
        },
        "question_results": [],
        "errors": [],
        "summary": {
            "majority_voting": {"correct": 0, "total": 0},
            "weighted_voting": {"correct": 0, "total": 0},
            "borda_count": {"correct": 0, "total": 0}
        },
        "disagreement_summary": {
            "total_disagreements": 0,
            "disagreement_rate": 0.0,
            "disagreement_patterns": {}
        },
        "medrag_summary": {
            "enabled": use_medrag,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_snippets_retrieved": 0,
            "average_retrieval_time": 0.0
        },
        "complexity_distribution": {"basic": 0, "intermediate": 0, "advanced": 0},
        "deployment_usage": {d['name']: 0 for d in deployments}
    }
    
    # Add summary for yes/no/maybe if PubMedQA
    if dataset_type == "pubmedqa":
        results["summary"]["yes_no_maybe_voting"] = {"correct": 0, "total": 0}
    
    # Process questions in parallel batches
    batch_size = num_deployments
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    with tqdm(total=len(questions), desc=f"{config_name}") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            
            # Prepare futures for this batch
            future_to_info = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_questions), num_deployments)) as executor:
                # Submit each question in the batch to a different deployment
                for i, question in enumerate(batch_questions):
                    question_index = batch_start + i
                    deployment_index = i % num_deployments  # Round-robin assignment
                    deployment_config = deployments[deployment_index]
                    
                    # Track deployment usage
                    results["deployment_usage"][deployment_config['name']] += 1
                    
                    # Submit the question processing
                    future = executor.submit(
                        process_single_question,
                        question_index,
                        question,
                        dataset_type,
                        configuration,
                        deployment_config,
                        run_output_dir,
                        max_retries,
                        use_medrag  
                    )
                    
                    future_to_info[future] = {
                        "question_index": question_index,
                        "deployment": deployment_config['name']
                    }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_info):
                    info = future_to_info[future]
                    try:
                        question_result = future.result()
                        results["question_results"].append(question_result)
                        

                        # Track MedRAG usage
                        if use_medrag and "medrag_info" in question_result:
                            medrag_info = question_result["medrag_info"]
                            if medrag_info.get("success", False):
                                results["medrag_summary"]["successful_retrievals"] += 1
                                results["medrag_summary"]["total_snippets_retrieved"] += medrag_info.get("snippets_retrieved", 0)
                                
                                # Track retrieval time
                                retrieval_time = medrag_info.get("retrieval_time", 0)
                                current_avg = results["medrag_summary"]["average_retrieval_time"]
                                current_count = results["medrag_summary"]["successful_retrievals"]
                                results["medrag_summary"]["average_retrieval_time"] = (
                                    (current_avg * (current_count - 1) + retrieval_time) / current_count
                                )
                            else:
                                results["medrag_summary"]["failed_retrievals"] += 1

                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'deployment': info['deployment'],
                            'processed': len(results["question_results"]),
                            'errors': len(results["errors"])
                        })
                        
                        # Update summary statistics if no error
                        if "error" not in question_result and "performance" in question_result:
                            task_performance = question_result["performance"]
                            
                            # Handle different task types properly
                            if dataset_type == "pubmedqa":
                                # For PubMedQA, all methods should work with yes/no/maybe
                                for method in ["majority_voting", "weighted_voting", "borda_count"]:
                                    if method in task_performance:
                                        method_perf = task_performance[method]
                                        if "correct" in method_perf:
                                            results["summary"][method]["total"] += 1
                                            if method_perf["correct"]:
                                                results["summary"][method]["correct"] += 1
                                
                            else:
                                methods_to_check = ["majority_voting", "weighted_voting", "borda_count"]
                                for method in methods_to_check:
                                    if method in task_performance:
                                        method_perf = task_performance[method]
                                        if "correct" in method_perf:
                                            results["summary"][method]["total"] += 1
                                            if method_perf["correct"]:
                                                results["summary"][method]["correct"] += 1
                        
                        # Track disagreements
                        if question_result.get("disagreement_flag", False):
                            results["disagreement_summary"]["total_disagreements"] += 1
                            
                            # Track disagreement patterns
                            disagreement_analysis = question_result.get("disagreement_analysis", {})
                            pattern = disagreement_analysis.get("answer_distribution", {})
                            pattern_key = "-".join(sorted(pattern.keys()))
                            if pattern_key:
                                results["disagreement_summary"]["disagreement_patterns"][pattern_key] = \
                                    results["disagreement_summary"]["disagreement_patterns"].get(pattern_key, 0) + 1
                        
                        # Track complexity
                        complexity = question_result.get("recruitment_info", {}).get("complexity_selected")
                        if complexity and complexity in results["complexity_distribution"]:
                            results["complexity_distribution"][complexity] += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing question {info['question_index']}: {str(e)}")
                        results["errors"].append({
                            "question_index": info['question_index'],
                            "deployment": info['deployment'],
                            "error_type": "processing_error",
                            "error": str(e)
                        })
                        pbar.update(1)
    
    # Calculate final statistics
    total_processed = len([q for q in results["question_results"] if "error" not in q])
    results["disagreement_summary"]["disagreement_rate"] = (
        results["disagreement_summary"]["total_disagreements"] / total_processed 
        if total_processed > 0 else 0
    )
    
    # Calculate accuracy for each method
    for method in results["summary"].keys():
        method_summary = results["summary"][method]
        method_summary["accuracy"] = (
            method_summary["correct"] / method_summary["total"] 
            if method_summary["total"] > 0 else 0.0
        )

    # Calculate mind change summary
    total_mind_changes = sum(
        q.get("mind_change_analysis", {}).get("agents_who_changed", 0) 
        for q in results["question_results"]
    )
    results["disagreement_summary"]["mind_change_summary"] = {
        "total_mind_changes": total_mind_changes,
        "average_change_rate": total_mind_changes / (total_processed * 5) if total_processed > 0 else 0
    }

    # Add complexity metrics
    from components import agent_recruitment
    if hasattr(agent_recruitment, "complexity_counts") and hasattr(agent_recruitment, "complexity_correct"):
        results["complexity_metrics"] = {
            "counts": agent_recruitment.complexity_counts.copy(),
            "correct": agent_recruitment.complexity_correct.copy(),
            "accuracy": {}
        }
        
        for level in ["basic", "intermediate", "advanced"]:
            count = results["complexity_metrics"]["counts"].get(level, 0)
            correct = results["complexity_metrics"]["correct"].get(level, 0)
            results["complexity_metrics"]["accuracy"][level] = correct / count if count > 0 else 0.0
    
    # Save enhanced overall results
    if run_output_dir:
        try:
            with open(os.path.join(run_output_dir, "summary.json"), 'w') as f:
                json.dump(results["summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "errors.json"), 'w') as f:
                json.dump(results["errors"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "disagreement_analysis.json"), 'w') as f:
                json.dump(results["disagreement_summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "deployment_usage.json"), 'w') as f:
                json.dump(results["deployment_usage"], f, indent=2)
                
            # Save comprehensive detailed results
            with open(os.path.join(run_output_dir, "detailed_results_enhanced.json"), 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save enhanced summary results: {str(e)}")
    
    # Enhanced summary print
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")

    if use_medrag:
        medrag_summary = results["medrag_summary"]
        total_attempts = medrag_summary["successful_retrievals"] + medrag_summary["failed_retrievals"]
        success_rate = medrag_summary["successful_retrievals"] / total_attempts if total_attempts > 0 else 0
        
        print(f"  MedRAG Enhancement:")
        print(f"    Success Rate: {medrag_summary['successful_retrievals']}/{total_attempts} ({success_rate:.2%})")
        print(f"    Total Snippets Retrieved: {medrag_summary['total_snippets_retrieved']}")
        print(f"    Average Retrieval Time: {medrag_summary['average_retrieval_time']:.2f}s")
    
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")
    
    print(f"  Disagreements: {results['disagreement_summary']['total_disagreements']}/{total_processed} questions ({results['disagreement_summary']['disagreement_rate']:.2%})")
    
    # Print complexity distribution
    if any(results["complexity_distribution"].values()):
        print(f"  Complexity Distribution: {dict(results['complexity_distribution'])}")
    
    # Print deployment usage
    print(f"  Deployment Usage: {dict(results['deployment_usage'])}")
        
    return results

def create_comprehensive_combined_results(all_results, dataset_type, output_dir):
    """Create comprehensive combined results with all enhanced data"""
    
    combined_results = {
        "dataset": dataset_type,
        "timestamp": datetime.now().isoformat(),
        "configurations_tested": len(all_results),
        "total_questions_per_config": all_results[0]["num_questions"] if all_results else 0,
        
        # Configuration summaries
        "configuration_summaries": {},
        
        # Cross-configuration analysis
        "cross_analysis": {
            "disagreement_comparison": {},
            "complexity_distribution": {},
            "agent_performance_patterns": {},
            "teamwork_effectiveness": {}
        },
        
        # All detailed results
        "detailed_results": all_results,
        
        # Meta-analysis
        "meta_analysis": {
            "best_performing_configs": {},
            "most_disagreement_prone": {},
            "complexity_accuracy_correlation": {},
            "agent_type_effectiveness": {}
        }
    }
    
    # Process each configuration's results
    for result in all_results:
        config_name = result["configuration"]
        
        # Configuration summary
        combined_results["configuration_summaries"][config_name] = {
            "accuracy_by_method": result["summary"],
            "disagreement_rate": result.get("disagreement_summary", {}).get("disagreement_rate", 0),
            "total_errors": len(result.get("errors", [])),
            "complexity_distribution": analyze_complexity_distribution(result),
            "agent_patterns": analyze_agent_patterns(result),
            "deployment_usage": result.get("deployment_usage", {})
        }
        
        # Cross-analysis data
        combined_results["cross_analysis"]["disagreement_comparison"][config_name] = \
            result.get("disagreement_summary", {})
    
    # Perform meta-analysis
    combined_results["meta_analysis"] = perform_meta_analysis(all_results)
    
    # Save comprehensive results
    if output_dir:
        with open(os.path.join(output_dir, "comprehensive_combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
    
    return combined_results

def analyze_complexity_distribution(result):
    """Analyze complexity distribution for a configuration"""
    complexity_counts = {"basic": 0, "intermediate": 0, "advanced": 0}
    complexity_accuracy = {"basic": {"correct": 0, "total": 0}, 
                          "intermediate": {"correct": 0, "total": 0}, 
                          "advanced": {"correct": 0, "total": 0}}
    
    for q_result in result.get("question_results", []):
        recruitment_info = q_result.get("recruitment_info", {})
        complexity = recruitment_info.get("complexity_selected", "unknown")
        
        if complexity in complexity_counts:
            complexity_counts[complexity] += 1
            
            # Check accuracy
            performance = q_result.get("performance", {})
            if "majority_voting" in performance and "correct" in performance["majority_voting"]:
                complexity_accuracy[complexity]["total"] += 1
                if performance["majority_voting"]["correct"]:
                    complexity_accuracy[complexity]["correct"] += 1
    
    return {"counts": complexity_counts, "accuracy": complexity_accuracy}

def analyze_agent_patterns(result):
    """Analyze agent recruitment and performance patterns"""
    agent_type_counts = {}
    agent_performance = {}
    
    for q_result in result.get("question_results", []):
        recruitment_info = q_result.get("recruitment_info", {})
        agents = recruitment_info.get("agents_recruited", [])
        
        for agent in agents:
            role = agent.get("role", "unknown")
            agent_type_counts[role] = agent_type_counts.get(role, 0) + 1
    
    return {"agent_type_distribution": agent_type_counts}

def perform_meta_analysis(all_results):
    """Perform meta-analysis across all configurations"""
    
    # Find best performing configurations
    best_configs = {}
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        best_accuracy = 0
        best_config = None
        
        for result in all_results:
            accuracy = result["summary"].get(method, {}).get("accuracy", 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = result["configuration"]
        
        best_configs[method] = {"config": best_config, "accuracy": best_accuracy}
    
    # Find most disagreement-prone configurations
    disagreement_ranking = []
    for result in all_results:
        disagreement_rate = result.get("disagreement_summary", {}).get("disagreement_rate", 0)
        disagreement_ranking.append({
            "config": result["configuration"],
            "disagreement_rate": disagreement_rate
        })
    
    disagreement_ranking.sort(key=lambda x: x["disagreement_rate"], reverse=True)
    
    return {
        "best_performing_configs": best_configs,
        "disagreement_ranking": disagreement_ranking
    }

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
    n_max: int = 5,
    use_medrag: bool = False
) -> Dict[str, Any]:
    """
    Run a dataset through the agent system with question-level parallel processing.
    
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
    
    # Log parameters including MedRAG
    medrag_status = "with MedRAG" if use_medrag else "without MedRAG"
    logging.info(f"Running dataset: {dataset_type} {medrag_status}, n_max={n_max}, recruitment_method={recruitment_method}")

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
    
    # Log MedRAG configuration
    if use_medrag:
        # Test MedRAG availability before processing
        test_integration = create_medrag_integration()
        if test_integration and test_integration.is_available():
            logging.info("MedRAG integration is available and will be used")
        else:
            error_msg = test_integration.get_initialization_error() if test_integration else "Failed to create integration"
            logging.warning(f"MedRAG integration not available: {error_msg}")
            logging.warning("Continuing without MedRAG enhancement")
            use_medrag = False
    
    # Setup output directory
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"{dataset_type}_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
    else:
        run_output_dir = None
    
    # Log deployment configuration
    available_deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in available_deployments]}")
    logging.info(f"Questions will be distributed across {len(available_deployments)} deployments in round-robin fashion")
    
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
                "n_max": 1,  # Always use just 1 agent for Baseline
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
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
            "n_max": n_max,  # Use specified n_max value
            "medrag": use_medrag  # Pass MedRAG usage
        }]
    
    # Run each configuration
    all_results = []
    for config_dict in configurations:
        # Ensure each config has proper n_max value
        if "n_max" not in config_dict:
            config_dict["n_max"] = n_max

        # Get MedRAG setting for this configuration
        config_use_medrag = config_dict.get("use_medrag", False)
        
        # Special handling for Baseline - always use basic recruitment with 1 agent
        if config_dict["name"] == "Baseline":
            config_dict["recruitment"] = True
            config_dict["recruitment_method"] = "basic"
            config_dict["n_max"] = 1
        
        # Add recruitment settings to all configs if recruitment is enabled
        if config_dict["recruitment"]:
            config_dict["recruitment_method"] = config_dict.get("recruitment_method", recruitment_method)
            config_dict["recruitment_pool"] = config_dict.get("recruitment_pool", recruitment_pool)
        
        # Log current configuration with MedRAG status
        description = config_dict.get("description", "")
        medrag_note = " with MedRAG" if config_use_medrag else ""
        desc_str = f" - {description}{medrag_note}" if description else medrag_note
        logging.info(f"Running configuration: {config_dict['name']}{desc_str}, recruitment={config_dict['recruitment']}, method={config_dict['recruitment_method']}, n_max={config_dict['n_max']}")
        
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config_dict,
            run_output_dir,
            n_max=config_dict.get("n_max", n_max),
             use_medrag=config_use_medrag 
        )
        all_results.append(result)
    
    # Compile combined results
    combined_results = {
        "dataset": dataset_type,
        "num_questions": num_questions,
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
        "parallel_processing": "question_level",
        "deployments_used": [d['name'] for d in available_deployments],
        "configurations": [r["configuration"] for r in all_results],
        "summaries": {r["configuration"]: r["summary"] for r in all_results},
        "medrag_summaries": {r["configuration"]: r.get("medrag_summary", {}) for r in all_results if r.get("use_medrag", False)}
    }
    
    # Save combined results
    if run_output_dir:
        with open(os.path.join(run_output_dir, "combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Also create comprehensive combined results
        create_comprehensive_combined_results(all_results, dataset_type, run_output_dir)
    
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
    
    # Add MedRAG argument
    parser.add_argument('--medrag', action='store_true', 
                      help='Enable MedRAG knowledge enhancement')
    parser.add_argument('--medrag-retriever', type=str, default='MedCPT',
                      help='MedRAG retriever to use (default: MedCPT)')
    parser.add_argument('--medrag-corpus', type=str, default='Textbooks',
                      help='MedRAG corpus to use (default: Textbooks)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Log MedRAG configuration if enabled
    if args.medrag:
        logging.info(f"MedRAG enhancement enabled with {args.medrag_retriever}/{args.medrag_corpus}")
        
        # Test MedRAG availability
        test_integration = create_medrag_integration(
            retriever_name=args.medrag_retriever,
            corpus_name=args.medrag_corpus
        )
        
        if test_integration and test_integration.is_available():
            logging.info("MedRAG integration test successful")
        else:
            error_msg = test_integration.get_initialization_error() if test_integration else "Failed to create integration"
            logging.error(f"MedRAG integration test failed: {error_msg}")
            logging.error("Consider disabling MedRAG or fixing the configuration")
            
            # Option to continue without MedRAG
            response = input("Continue without MedRAG? (y/n): ")
            if response.lower() != 'y':
                return
            args.medrag = False

    # Log deployment configuration
    deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in deployments]}")
    logging.info(f"Question-level parallel processing will be used with {len(deployments)} deployments")

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
        n_max=args.n_max,
        use_medrag=args.medrag  
    )
    
    # Print overall summary
    print("\nOverall Results:")
    for config_name, summary in results.get("summaries", {}).items():
        print(f"\n{config_name}:")
        for method, stats in summary.items():
            if "accuracy" in stats:
                print(f"  {method.replace('_', ' ').title()}: {stats['accuracy']:.2%} accuracy")

    # Print MedRAG summary if used
    if args.medrag and "medrag_summaries" in results:
        print(f"\nMedRAG Enhancement Summary:")
        for config_name, medrag_summary in results["medrag_summaries"].items():
            if medrag_summary:
                total_attempts = medrag_summary.get("successful_retrievals", 0) + medrag_summary.get("failed_retrievals", 0)
                success_rate = medrag_summary.get("successful_retrievals", 0) / total_attempts if total_attempts > 0 else 0
                print(f"  {config_name}:")
                print(f"    Retrieval Success Rate: {success_rate:.2%}")
                print(f"    Average Retrieval Time: {medrag_summary.get('average_retrieval_time', 0):.2f}s")
                print(f"    Total Snippets: {medrag_summary.get('total_snippets_retrieved', 0)}")

    # Print deployment information
    deployments = config.get_all_deployments()
    print(f"\nDeployment Information:")
    print(f"  Total deployments used: {len(deployments)}")
    print(f"  Deployment names: {[d['name'] for d in deployments]}")
    print(f"  Processing method: Question-level parallel (each question assigned to a deployment)")

if __name__ == "__main__":
    main()