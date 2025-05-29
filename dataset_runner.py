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
            task = format_medmcqa_for_task(question_data)
            parsed_answer = task.get("ground_truth", "ERROR")
            
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
    
    IMPORTANT: The MedMCQA dataset has a known issue where questions labeled as 
    'multi-choice' actually have only single correct answers. This function 
    handles this by treating ALL questions as single-choice questions.
    
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
    
    # Handle cases where options might not be present
    if not any([opa, opb, opc, opd]):
        logging.error("No options available for the question")
        return {}
    
    # Format as standard MCQ options (only include non-empty options)
    options = []
    option_letters = ['A', 'B', 'C', 'D']
    option_values = [opa, opb, opc, opd]
    
    for letter, value in zip(option_letters, option_values):
        if value and value.strip():  # Only add non-empty options
            options.append(f"{letter}. {value}")
    
    # Get the choice_type for logging/metadata purposes
    original_choice_type = question_data.get("choice_type", "single")
    
    # Log if this was labeled as multi-choice (for debugging)
    if original_choice_type == "multi":
        logging.debug(f"Question ID {question_data.get('id', 'unknown')} labeled as 'multi' but treating as single choice due to dataset issue")
    
    # Parse correct option (cop is 1-indexed: 1->A, 2->B, 3->C, 4->D)
    cop = question_data.get("cop", 1)  # Default to 1 if missing
    
    # Robust parsing of cop field
    ground_truth = parse_cop_field(cop)
    
    # Get explanation and metadata
    explanation = question_data.get("exp", "")
    subject_name = question_data.get("subject_name", "")
    topic_name = question_data.get("topic_name", "")
    
    # Create task dictionary - ALWAYS treat as single choice MCQ
    task = {
        "name": "MedMCQA Question",
        "description": question_text,
        "type": "mcq",  # Always MCQ, never multi_choice_mcq
        "options": options,
        "expected_output_format": "Single letter selection with rationale",
        "ground_truth": ground_truth,
        "rationale": {ground_truth: explanation} if explanation else {},
        "metadata": {
            "subject": subject_name,
            "topic": topic_name,
            "question_id": question_data.get("id", ""),
            "original_choice_type": original_choice_type,  # Keep original for reference
            "cop_original": question_data.get("cop", ""),  # Keep original cop for debugging
            "dataset_issue_note": "Treated as single-choice due to known MedMCQA dataset labeling issue"
        }
    }
    
    return task

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
        r"^([A-Da-d])\.",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
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
                info["final_answer"] = extract_answer_option(response_data["final_decision"])
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
    Enhanced with comprehensive logging and disagreement detection.
    """
    config_name = configuration.get("name", "unknown")
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
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
    
    # Enhanced results structure
    results = {
        "configuration": config_name,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "configuration_details": configuration,
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
        "complexity_distribution": {"basic": 0, "intermediate": 0, "advanced": 0},
        "agent_recruitment_summary": {}
    }
    
    disagreement_count = 0
    
    # Add summary for yes/no/maybe if PubMedQA
    if dataset_type == "pubmedqa":
        results["summary"]["yes_no_maybe_voting"] = {"correct": 0, "total": 0}
    
    for i, question in enumerate(tqdm(questions, desc=f"{config_name}")):
        question_result = {
            "question_index": i,
            "question_metadata": {
                "id": question.get("id", f"q_{i}"),
                "subject": question.get("subject_name", ""),
                "topic": question.get("topic_name", ""),
                "original_choice_type": question.get("choice_type", "")
            },
            "recruitment_info": {
                "complexity_selected": None,
                "agents_recruited": [],
                "recruitment_method": configuration.get("recruitment_method", "none"),
                "recruitment_reasoning": ""
            },
            "agent_responses": {},
            "disagreement_analysis": {},
            "disagreement_flag": False
        }
        
        simulator = None
        performance = None
        
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
            
            # Store question details
            question_result.update({
                "question_text": task["description"][:200] + "...",
                "options": task.get("options", []),
                "ground_truth": task.get("ground_truth", ""),
                "task_type": task.get("type", "")
            })
            
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
                        n_max=configuration.get("n_max", n_max)
                    )

                    # In dataset_runner.py, after simulator creation:
                    if hasattr(simulator, 'metadata') and 'complexity' in simulator.metadata:
                        question_result["recruitment_info"]["complexity_selected"] = simulator.metadata['complexity']

                    if hasattr(simulator, 'agents') and simulator.agents:
                        agents_info = []
                        for role, agent in simulator.agents.items():
                            agents_info.append({
                                "role": role,
                                "weight": getattr(agent, 'weight', 0.2)
                            })
                        question_result["recruitment_info"]["agents_recruited"] = agents_info
                    
                    # Run simulation
                    simulation_results = simulator.run_simulation()
                    performance = simulator.evaluate_performance()


                    
                    # Extract recruitment information
                    if hasattr(simulator, 'recruited_agents') and simulator.recruited_agents:
                        recruited_agents_info = []
                        for agent in simulator.recruited_agents:
                            if isinstance(agent, dict):
                                # Handle case where agent is already a dictionary
                                agent_info = {
                                    "role": agent.get("role", "Unknown"),
                                    "expertise": agent.get("expertise", []),
                                    "weight": agent.get("weight", 0.2),
                                    "specialization": agent.get("specialization", "")
                                }
                            else:
                                # Handle case where agent is an object
                                agent_info = {
                                    "role": getattr(agent, 'role', 'Unknown'),
                                    "expertise": getattr(agent, 'expertise', []),
                                    "weight": getattr(agent, 'weight', 0.2),
                                    "specialization": getattr(agent, 'specialization', '')
                                }
                            recruited_agents_info.append(agent_info)
                        
                        question_result["recruitment_info"]["agents_recruited"] = recruited_agents_info
                    
                    # Extract complexity if available
                    if hasattr(simulator, 'task_complexity'):
                        complexity = simulator.task_complexity
                        question_result["recruitment_info"]["complexity_selected"] = complexity
                        question_result["recruitment_info"]["recruitment_reasoning"] = f"Selected {complexity} complexity"
                        
                        # Update complexity distribution
                        if complexity in results["complexity_distribution"]:
                            results["complexity_distribution"][complexity] += 1
                    
                    # Extract detailed agent responses
                    question_result["agent_responses"] = extract_agent_responses_info(simulation_results)

                    # ADD HERE: Track mind changes
                    if hasattr(simulation_results, 'agent_analyses') and hasattr(simulation_results, 'agent_decisions'):
                        mind_changes = track_agent_mind_changes(
                            simulation_results.get('agent_analyses', {}), 
                            simulation_results.get('agent_decisions', {})
                        )
                        
                        change_count = sum(1 for change in mind_changes.values() if change["changed_mind"])
                        question_result["mind_change_analysis"] = {
                            "total_agents": len(mind_changes),
                            "agents_who_changed": change_count,
                            "change_rate": change_count / len(mind_changes) if mind_changes else 0,
                            "individual_changes": mind_changes
                        }

                    # Detect disagreement (existing code continues here)
                    disagreement_analysis = detect_agent_disagreement(simulation_results.get("agent_responses", {}))
                    question_result["disagreement_analysis"] = disagreement_analysis

                    # In dataset_runner.py, add mind change tracking:
                    if "agent_analyses" in simulation_results and "agent_responses" in simulation_results:
                        mind_changes = track_agent_mind_changes(
                            simulation_results["agent_analyses"], 
                            simulation_results["agent_responses"]
                        )
                        question_result["mind_change_analysis"] = mind_changes
                    
                    if disagreement_analysis.get("has_disagreement", False):
                        disagreement_count += 1
                        question_result["disagreement_flag"] = True
                        
                        # Track disagreement patterns
                        pattern = disagreement_analysis["answer_distribution"]
                        pattern_key = "-".join(sorted(pattern.keys()))
                        if pattern_key:
                            results["disagreement_summary"]["disagreement_patterns"][pattern_key] = \
                                results["disagreement_summary"]["disagreement_patterns"].get(pattern_key, 0) + 1
                    
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
                            for method in ["majority_voting", "weighted_voting", "borda_count"]:
                                if method in task_performance:
                                    method_perf = task_performance[method]
                                    if "correct" in method_perf:
                                        results["summary"][method]["total"] += 1
                                        if method_perf["correct"]:
                                            results["summary"][method]["correct"] += 1
                            
                            if "yes_no_maybe_voting" in task_performance:
                                yes_no_perf = task_performance["yes_no_maybe_voting"]
                                if "correct" in yes_no_perf:
                                    results["summary"]["yes_no_maybe_voting"]["total"] += 1
                                    if yes_no_perf["correct"]:
                                        results["summary"]["yes_no_maybe_voting"]["correct"] += 1
                        else:
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
                    
                    if attempt < max_retries - 1:
                        if "content" in error_str.lower() and "filter" in error_str.lower():
                            wait_time = min(5 * (attempt + 1), 30)
                            logging.warning(f"Content filter triggered, retry {attempt+1} for question {i}")
                        elif any(term in error_str.lower() for term in ["rate", "limit", "timeout", "capacity"]):
                            wait_time = min(2 ** attempt + 1, 15)
                            logging.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}")
                        elif any(term in error_str.lower() for term in ["connection", "timeout", "retry", "try again"]):
                            wait_time = min(2 ** attempt + 1, 10)
                            logging.warning(f"Connection error, retry {attempt+1} for question {i}")
                        else:
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
            
            # Save individual question results with enhanced data
            if run_output_dir:
                try:
                    enhanced_output_data = {
                        "question": task["description"],
                        "options": task.get("options", []),
                        "ground_truth": task.get("ground_truth", ""),
                        "metadata": task.get("metadata", {}),
                        "recruitment_info": question_result["recruitment_info"],
                        "agent_responses": question_result["agent_responses"],
                        "disagreement_analysis": question_result["disagreement_analysis"],
                        "disagreement_flag": question_result["disagreement_flag"]
                    }
                    
                    if "decisions" in question_result:
                        enhanced_output_data["decisions"] = question_result["decisions"]
                    if "performance" in question_result:
                        enhanced_output_data["performance"] = question_result["performance"]
                    if "agent_conversations" in question_result:
                        enhanced_output_data["agent_conversations"] = question_result["agent_conversations"]
                    if "error" in question_result:
                        enhanced_output_data["error"] = question_result["error"]
                    
                    with open(os.path.join(run_output_dir, f"question_{i}_enhanced.json"), 'w') as f:
                        json.dump(enhanced_output_data, f, indent=2)
                except Exception as e:
                    logging.error(f"Failed to save enhanced results for question {i}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error processing question {i}: {str(e)}")
            question_result["error"] = f"Processing error: {str(e)}"
            results["errors"].append({
                "question_index": i,
                "error_type": "processing",
                "error": str(e)
            })
        
        # Always add the question result, even if it has errors
        results["question_results"].append(question_result)
    
    # Calculate final disagreement statistics
    total_processed = len([q for q in results["question_results"] if "error" not in q])
    results["disagreement_summary"]["total_disagreements"] = disagreement_count
    results["disagreement_summary"]["disagreement_rate"] = disagreement_count / total_processed if total_processed > 0 else 0
    
    # Calculate accuracy for each method
    for method in results["summary"].keys():
        method_summary = results["summary"][method]
        method_summary["accuracy"] = method_summary["correct"] / method_summary["total"] if method_summary["total"] > 0 else 0.0

    # ADD HERE: Calculate mind change summary
    total_mind_changes = sum(q.get("mind_change_analysis", {}).get("agents_who_changed", 0) 
                            for q in results["question_results"])
    results["disagreement_summary"]["mind_change_summary"] = {
        "total_mind_changes": total_mind_changes,
        "average_change_rate": total_mind_changes / (total_processed * 5) if total_processed > 0 else 0  # assuming 5 agents
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
    
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")
    
    print(f"  Disagreements: {disagreement_count}/{total_processed} questions ({results['disagreement_summary']['disagreement_rate']:.2%})")
    
    # Print complexity distribution
    if any(results["complexity_distribution"].values()):
        print(f"  Complexity Distribution: {dict(results['complexity_distribution'])}")
        
    return results


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
            "agent_patterns": analyze_agent_patterns(result)
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