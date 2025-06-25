"""
Base agent class for modular agent system with isolated task configuration support.
"""

import logging
import json
import re
import time
import signal
import threading
from typing import List, Dict, Any, Optional, Tuple
import copy

from langchain_openai import AzureChatOpenAI
import config

from utils.prompts import (
    AGENT_SYSTEM_PROMPTS,
    LEADERSHIP_PROMPTS,
    COMMUNICATION_PROMPTS,
    MONITORING_PROMPTS,
    MENTAL_MODEL_PROMPTS,
    ORIENTATION_PROMPTS,
    TRUST_PROMPTS
)


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


class Agent:
    """Base agent class for modular agent system with isolated task configuration support."""
    
    def __init__(self, 
                 role: str, 
                 expertise_description: str,
                 use_team_leadership: bool = False,
                 use_closed_loop_comm: bool = False,
                 use_mutual_monitoring: bool = False,
                 use_shared_mental_model: bool = False,
                 use_team_orientation: bool = False,
                 use_mutual_trust: bool = False,
                 n_max: int = 5,
                 examples: Optional[List[Dict[str, str]]] = None,
                 deployment_config: Optional[Dict[str, str]] = None,
                 agent_index: int = 0):
        """
        Initialize an LLM-based agent with a specific role.
        
        Args:
            role: The role of the agent (e.g., "Critical Analyst")
            expertise_description: Description of the agent's expertise
            use_team_leadership: Whether this agent uses team leadership behaviors
            use_closed_loop_comm: Whether this agent uses closed-loop communication
            use_mutual_monitoring: Whether this agent uses mutual performance monitoring
            use_shared_mental_model: Whether this agent uses shared mental models
            use_team_orientation: Whether this agent uses team orientation
            use_mutual_trust: Whether this agent uses mutual trust
            n_max: Maximum number of agents
            examples: Optional examples to include in the prompt
            deployment_config: Optional specific deployment configuration
            agent_index: Index of agent for deployment assignment
        """
        self.role = role
        self.expertise_description = expertise_description
        self.use_team_leadership = use_team_leadership
        self.use_closed_loop_comm = use_closed_loop_comm
        self.use_mutual_monitoring = use_mutual_monitoring
        self.use_shared_mental_model = use_shared_mental_model
        self.use_team_orientation = use_team_orientation
        self.use_mutual_trust = use_mutual_trust
        self.n_max = n_max
        self.examples = examples or []
        self.conversation_history = []
        self.knowledge_base = {}
        self.agent_index = agent_index
        
        # Initialize logger
        self.logger = logging.getLogger(f"agent.{role}")

        # Get deployment configuration
        if deployment_config:
            self.deployment_config = deployment_config
        else:
            self.deployment_config = config.get_deployment_for_agent(agent_index)
        
        # Initialize LLM with specific deployment
        self.client = AzureChatOpenAI(
            azure_deployment=self.deployment_config["deployment"],
            api_key=self.deployment_config["api_key"],
            api_version=self.deployment_config["api_version"],
            azure_endpoint=self.deployment_config["endpoint"],
            temperature=config.TEMPERATURE,
            timeout=config.REQUEST_TIMEOUT
        )
        
        # Build initial system message using global config (for backward compatibility)
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add example conversations if provided
        if self.examples:
            for example in self.examples:
                self.messages.append({"role": "user", "content": example['question']})
                self.messages.append({
                    "role": "assistant", 
                    "content": example['answer'] + "\n\n" + example.get('reason', '')
                })
                
        self.logger.info(f"Initialized {self.role} agent with deployment {self.deployment_config['name']}")
    
    def _build_system_prompt(self, task_config: Dict[str, Any] = None) -> str:
        """Build the system prompt for the agent using isolated or global task config."""
        # Use provided task config or fall back to global
        task_config = task_config or config.TASK
        
        # Base prompt
        prompt = AGENT_SYSTEM_PROMPTS["base"].format(
            role=self.role,
            expertise_description=self.expertise_description,
            team_name=config.TEAM_NAME,
            team_goal=config.TEAM_GOAL,
            task_name=task_config['name'],
            task_description=task_config['description'],
            task_type=task_config['type'],
            expected_output_format=task_config.get('expected_output_format', 'not specified')
        )

        # Add team leadership component if enabled
        if self.use_team_leadership:
            prompt += LEADERSHIP_PROMPTS["team_leadership"]
        
        # Add closed-loop communication component if enabled
        if self.use_closed_loop_comm:
            prompt += COMMUNICATION_PROMPTS["closed_loop"]
            
        # Add mutual performance monitoring if enabled
        if self.use_mutual_monitoring:
            prompt += MONITORING_PROMPTS["mutual_monitoring"]
            
        # Add shared mental model if enabled
        if self.use_shared_mental_model:
            prompt += MENTAL_MODEL_PROMPTS["shared_mental_model"]
        
        # Add team orientation if enabled
        if self.use_team_orientation:
            prompt += ORIENTATION_PROMPTS["team_orientation"]
        
        # Add mutual trust if enabled
        if self.use_mutual_trust:
            # Base mutual trust prompt
            prompt += TRUST_PROMPTS["mutual_trust_base"]
            
            # Adjust for trust levels
            trust_factor = getattr(self, 'mutual_trust_factor', 0.8)
            if trust_factor < 0.4:
                prompt += TRUST_PROMPTS["low_trust"]
            elif trust_factor > 0.7:
                prompt += TRUST_PROMPTS["high_trust"]
        
        return prompt

    def add_to_knowledge_base(self, key: str, value: Any) -> None:
        """
        Add information to the agent's knowledge base.
        
        Args:
            key: The key for the knowledge
            value: The value of the knowledge
        """
        self.knowledge_base[key] = value
        self.logger.info(f"Added to knowledge base: {key}")
    
    def get_from_knowledge_base(self, key: str) -> Any:
        """
        Retrieve information from the agent's knowledge base.
        
        Args:
            key: The key for the knowledge
            
        Returns:
            The value of the knowledge, or None if not found
        """
        return self.knowledge_base.get(key)
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        raise TimeoutError("Request timed out")
    
    def _chat_with_timeout(self, messages: List[Dict[str, str]], timeout: int = None) -> str:
        """
        Chat with timeout and retry logic.
        
        Args:
            messages: Messages to send
            timeout: Timeout in seconds
            
        Returns:
            Assistant's response
            
        Raises:
            TimeoutError: If request times out
            Exception: If all retries fail
        """
        if timeout is None:
            timeout = config.INACTIVITY_TIMEOUT
        
        def target():
            try:
                response = self.client.invoke(messages)
                self._response = response.content
            except TypeError as e:
                if "missing 1 required positional argument: 'input'" in str(e):
                    # Fall back to old style invocation
                    response = self.client.invoke(input=messages)
                    self._response = response.content
                else:
                    raise
            except Exception as e:
                self._exception = e
        
        # Reset response and exception
        self._response = None
        self._exception = None
        
        # Start thread for API call
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            self.logger.warning(f"Request timed out after {timeout} seconds")
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        
        if self._exception:
            raise self._exception
        
        if self._response is None:
            raise Exception("No response received")
        
        return self._response


    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response with MedRAG enhancement.
        FIXED VERSION - Integrates MedRAG knowledge into responses.
        
        Args:
            message: The message to send to the agent
            
        Returns:
            The agent's response
        """
        self.logger.info(f"Received message: {message[:100]}...")
        
        # CRITICAL FIX: Check for MedRAG enhancement and integrate knowledge
        enhanced_message = message
        if self.get_from_knowledge_base("has_medrag_enhancement"):
            medrag_context = self.get_from_knowledge_base("medrag_context")
            if medrag_context:
                enhanced_message = f"""{message}

    {medrag_context}

    Based on the retrieved medical literature above, provide your analysis considering this evidence alongside your clinical expertise. If the literature supports, contradicts, or adds important context to your reasoning, please integrate this into your response."""
                
                self.logger.info("Applied MedRAG knowledge enhancement to agent message")
        
        # Add the enhanced message to the conversation
        messages = self.messages + [{"role": "user", "content": enhanced_message}]
        
        last_exception = None
        
        for attempt in range(config.MAX_RETRIES):
            try:
                self.logger.info(f"API request attempt {attempt + 1}/{config.MAX_RETRIES} using {self.deployment_config['name']}")
                
                # Try with timeout
                assistant_message = self._chat_with_timeout(messages, config.INACTIVITY_TIMEOUT)
                
                # Success - store the response (use original message, not enhanced)
                self.messages.append({"role": "user", "content": message})
                self.messages.append({"role": "assistant", "content": assistant_message})
                self.conversation_history.append({"user": message, "assistant": assistant_message})
                
                self.logger.info(f"Responded successfully: {assistant_message[:100]}...")
                return assistant_message
                
            except TimeoutError as e:
                last_exception = e
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                
                if attempt < config.MAX_RETRIES - 1:
                    wait_time = config.RETRY_DELAY * (2 ** attempt)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                if any(term in error_str for term in ["rate", "limit", "timeout", "capacity", "connection"]):
                    self.logger.warning(f"Recoverable error on attempt {attempt + 1}: {str(e)}")
                    
                    if attempt < config.MAX_RETRIES - 1:
                        if "rate" in error_str or "limit" in error_str:
                            wait_time = config.RETRY_DELAY * (3 ** attempt)
                        else:
                            wait_time = config.RETRY_DELAY * (2 ** attempt)
                        
                        self.logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                else:
                    self.logger.error(f"Non-recoverable error: {str(e)}")
                    raise
        
        # All retries failed
        self.logger.error(f"All {config.MAX_RETRIES} attempts failed. Last error: {str(last_exception)}")
        raise Exception(f"Failed after {config.MAX_RETRIES} attempts. Last error: {str(last_exception)}")


    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def extract_multi_choice_mcq_answer(self, message):
        """
        Extract multi-choice MCQ answers from the agent's response.
        
        Args:
            message: Message to analyze
            
        Returns:
            Multi-choice MCQ answers and confidence level
        """
        import re
        
        # Patterns for multi-choice answers
        patterns = [
            r"ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"FINAL ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"selected options?:?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"correct options? (?:are|is):?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Extract and clean the answer string
                answer_str = match.group(1).upper()
                # Remove 'and', spaces, and split by comma
                answer_str = answer_str.replace('AND', ',').replace(' ', '')
                answers = [a.strip() for a in answer_str.split(',') if a.strip()]
                # Remove duplicates and sort
                answers = sorted(list(set(answers)))
                confidence = self.extract_confidence(message)
                return {"answers": answers, "confidence": confidence}
        
        # Look for pattern like "Options A and C" or "A, B, and D"
        option_pattern = r"options?\s+([A-D](?:\s*,\s*[A-D])*(?:\s*,?\s*and\s*[A-D])?)"
        match = re.search(option_pattern, message, re.IGNORECASE)
        if match:
            answer_str = match.group(1).upper()
            answer_str = answer_str.replace('AND', ',').replace(' ', '')
            answers = [a.strip() for a in answer_str.split(',') if a.strip()]
            answers = sorted(list(set(answers)))
            confidence = self.extract_confidence(message)
            return {"answers": answers, "confidence": confidence}
        
        # No clear answers found
        confidence = self.extract_confidence(message)
        return {"answers": [], "confidence": confidence}

    # Update extract_response method:
    def extract_response(self, message=None) -> Dict[str, Any]:
        """
        Extract the agent's response to the task from their message using global config.
        This is the legacy method for backward compatibility.
        
        Args:
            message: Optional message to analyze, defaults to last response
            
        Returns:
            Structured response based on task type
        """
        if message is None:
            if not self.conversation_history:
                return {}
            message = self.conversation_history[-1]["assistant"]
        
        return self.extract_response_isolated(message, config.TASK)
    
    def extract_response_isolated(self, message: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the agent's response to the task from their message using isolated task config.
        
        Args:
            message: Message to analyze
            task_config: Isolated task configuration
            
        Returns:
            Structured response based on task type
        """
        task_type = task_config["type"]
        
        if task_type == "ranking":
            return self.extract_ranking_isolated(message, task_config)
        elif task_type == "mcq":
            return self.extract_mcq_answer_isolated(message, task_config)
        elif task_type == "multi_choice_mcq":
            return self.extract_multi_choice_mcq_answer_isolated(message, task_config)
        elif task_type == "yes_no_maybe":
            return self.extract_yes_no_maybe_answer(message)
        elif task_type in ["open_ended", "estimation", "selection"]:
            return {"response": message, "confidence": self.extract_confidence(message)}
        else:
            return {"raw_response": message}

    def extract_ranking_isolated(self, message: str, task_config: Dict[str, Any]):
        """
        Extract a ranking from the agent's response using isolated task config.
        
        Args:
            message: Message to analyze
            task_config: Isolated task configuration
            
        Returns:
            List of ranked items and confidence level
        """
        ranking = []
        lines = message.split('\n')
        
        # Look for numbered items (1. Item, 2. Item, etc.)
        for line in lines:
            for i in range(1, len(task_config["options"]) + 1):
                if f"{i}." in line or f"{i}:" in line:
                    for item in task_config["options"]:
                        if item.lower() in line.lower():
                            ranking.append(item)
                            break
        
        # Check for duplicates and missing items
        seen_items = set()
        valid_ranking = []
        
        for item in ranking:
            if item not in seen_items:
                seen_items.add(item)
                valid_ranking.append(item)
        
        # Add any missing items at the end
        for item in task_config["options"]:
            if item not in seen_items:
                valid_ranking.append(item)
                seen_items.add(item)
        
        # Extract confidence level
        confidence = self.extract_confidence(message)
        
        return {
            "ranking": valid_ranking[:len(task_config["options"])],
            "confidence": confidence
        }
    
    def extract_ranking(self, message):
        """Legacy wrapper that uses global config.TASK."""
        return self.extract_ranking_isolated(message, config.TASK)
    
    def extract_mcq_answer_isolated(self, message: str, task_config: Dict[str, Any]):
        """
        Extract an MCQ answer from the agent's response using isolated task config.
        
        Args:
            message: Message to analyze
            task_config: Isolated task configuration
            
        Returns:
            MCQ answer and confidence level
        """
        # Look for option identifiers (A, B, C, D, etc.)
        for line in message.split('\n'):
            line = line.strip()
            for option in task_config["options"]:
                option_id = option.split('.')[0].strip() if '.' in option else None
                if option_id and (line.startswith(option_id) or f"Option {option_id}" in line or f"Answer: {option_id}" in line):
                    # Extract confidence level
                    confidence = self.extract_confidence(message)
                    return {"answer": option_id, "confidence": confidence}
        
        # Extract confidence level
        confidence = self.extract_confidence(message)
        
        # If no explicit option identifier is found, look for the full option text
        for line in message.split('\n'):
            for option in task_config["options"]:
                # Extract option identifier (A, B, C, etc.)
                option_id = option.split('.')[0].strip() if '.' in option else None
                # Check if the full option text is in the line
                if option[2:].strip().lower() in line.lower():
                    return {"answer": option_id, "confidence": confidence}
        
        # If still not found, try to determine if there's any indication of an answer
        lower_message = message.lower()
        for option in task_config["options"]:
            option_id = option.split('.')[0].strip() if '.' in option else None
            if option_id and f"select {option_id.lower()}" in lower_message or f"choose {option_id.lower()}" in lower_message:
                return {"answer": option_id, "confidence": confidence}
        
        # No clear answer found
        return {"answer": None, "confidence": confidence}
    
    def extract_mcq_answer(self, message):
        """Legacy wrapper that uses global config.TASK."""
        return self.extract_mcq_answer_isolated(message, config.TASK)
    
    def extract_multi_choice_mcq_answer_isolated(self, message: str, task_config: Dict[str, Any]):
        """
        Extract multi-choice MCQ answers from the agent's response using isolated task config.
        
        Args:
            message: Message to analyze
            task_config: Isolated task configuration
            
        Returns:
            Multi-choice MCQ answers and confidence level
        """
        import re
        
        # Patterns for multi-choice answers
        patterns = [
            r"ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"FINAL ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"selected options?:?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"correct options? (?:are|is):?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Extract and clean the answer string
                answer_str = match.group(1).upper()
                # Remove 'and', spaces, and split by comma
                answer_str = answer_str.replace('AND', ',').replace(' ', '')
                answers = [a.strip() for a in answer_str.split(',') if a.strip()]
                # Remove duplicates and sort
                answers = sorted(list(set(answers)))
                confidence = self.extract_confidence(message)
                return {"answers": answers, "confidence": confidence}
        
        # Look for pattern like "Options A and C" or "A, B, and D"
        option_pattern = r"options?\s+([A-D](?:\s*,\s*[A-D])*(?:\s*,?\s*and\s*[A-D])?)"
        match = re.search(option_pattern, message, re.IGNORECASE)
        if match:
            answer_str = match.group(1).upper()
            answer_str = answer_str.replace('AND', ',').replace(' ', '')
            answers = [a.strip() for a in answer_str.split(',') if a.strip()]
            answers = sorted(list(set(answers)))
            confidence = self.extract_confidence(message)
            return {"answers": answers, "confidence": confidence}
        
        # No clear answers found
        confidence = self.extract_confidence(message)
        return {"answers": [], "confidence": confidence}
    
    def _majority_voting_yes_no_maybe(self, agent_responses):
        """Apply majority voting to yes/no/maybe tasks."""
        votes = {"yes": 0, "no": 0, "maybe": 0}
        
        for agent_role, response_data in agent_responses.items():
            answer = None
            
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    answer = self.extract_yes_no_maybe_answer(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answer" in response_data["extract"]:
                        answer = response_data["extract"]["answer"]
                        if answer and answer.lower() in ["yes", "no", "maybe"]:
                            answer = answer.lower()
            elif isinstance(response_data, str):
                answer = self.extract_yes_no_maybe_answer(response_data)
            
            if answer and answer in votes:
                votes[answer] += 1
        
        # Find winning option
        total_votes = sum(votes.values())
        winning_option = max(votes, key=votes.get) if total_votes > 0 else None
        
        return {
            "method": "majority_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "total_votes": total_votes,
            "confidence": votes.get(winning_option, 0) / total_votes if total_votes > 0 else 0
        }

    # Fix 2: Update extract_yes_no_maybe_answer in decision_methods.py
    def extract_yes_no_maybe_answer(self, content):
        """Extract yes/no/maybe answer with improved parsing."""
        if not isinstance(content, str):
            return None
        
        content_lower = content.lower()
        
        # Explicit answer patterns - more comprehensive
        patterns = [
            r"ANSWER:\s*(yes|no|maybe)",
            r"^ANSWER:\s*(yes|no|maybe)",          # Start of line
            r"FINAL ANSWER:\s*(yes|no|maybe)",
            r"answer is:?\s*(yes|no|maybe)",
            r"my answer:?\s*(yes|no|maybe)",
            r"the answer is:?\s*(yes|no|maybe)",
            r"conclusion:?\s*(yes|no|maybe)",
            r"therefore:?\s*(yes|no|maybe)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).lower()
        
        # Check for clear yes/no/maybe statements at start of lines
        lines = content_lower.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line.startswith('yes,') or line.startswith('yes.') or line == 'yes':
                return "yes"
            elif line.startswith('no,') or line.startswith('no.') or line == 'no':
                return "no"
            elif line.startswith('maybe') or line.startswith('uncertain') or line.startswith('possibly'):
                return "maybe"
        
        return None

    # Fix 3: Update simulator.py _evaluate_yes_no_maybe_performance method
    def _evaluate_yes_no_maybe_performance(self) -> Dict[str, Any]:
        """Evaluate performance on yes/no/maybe tasks."""
        ground_truth = self.evaluation_data.get("ground_truth")
        if ground_truth:
            ground_truth = ground_truth.lower()
        
        if not ground_truth or ground_truth not in ["yes", "no", "maybe"]:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        for method, result in self.results["decision_results"].items():
            if result and "winning_option" in result:
                selected = result["winning_option"]
                if selected:
                    selected = selected.lower()
                correct = selected == ground_truth if selected else False
                
                metrics[method] = {
                    "correct": correct,
                    "confidence": result.get("confidence", 0),
                    "selected": selected,
                    "ground_truth": ground_truth
                }
            else:
                metrics[method] = {
                    "correct": False,
                    "confidence": 0,
                    "selected": None,
                    "ground_truth": ground_truth
                }
        
        return metrics
    
    def extract_confidence(self, message):
        """
        Extract a confidence level from the agent's response.
        
        Args:
            message: Message to analyze
            
        Returns:
            Confidence level (0.0-1.0)
        """
        confidence = 0.7  # Default medium-high confidence
        
        # Look for explicit confidence statements
        lower_message = message.lower()
        
        if "confidence: " in lower_message:
            # Try to extract numeric confidence
            confidence_parts = lower_message.split("confidence: ")[1].split()
            if confidence_parts:
                try:
                    # Handle percentage format
                    if "%" in confidence_parts[0]:
                        confidence_value = float(confidence_parts[0].replace("%", "")) / 100
                        confidence = min(max(confidence_value, 0.0), 1.0)
                    else:
                        # Handle decimal format
                        confidence_value = float(confidence_parts[0])
                        # If it's on a 0-10 scale, convert to 0-1
                        if confidence_value > 1.0:
                            confidence_value /= 10
                        confidence = min(max(confidence_value, 0.0), 1.0)
                except ValueError:
                    # If conversion fails, use linguistic markers
                    pass
        
        # Check for linguistic confidence markers if no explicit value found
        if confidence == 0.7:
            high_confidence_markers = ["certainly", "definitely", "absolutely", "strongly believe", "confident", "sure", "clear evidence", "conclusive"]
            medium_confidence_markers = ["likely", "probably", "think", "believe", "reasonable", "should be", "suggests", "indicates"]
            low_confidence_markers = ["uncertain", "might", "possibly", "guess", "not sure", "doubtful", "maybe", "unclear", "insufficient"]
            
            # Count markers in each category
            high_count = sum(1 for marker in high_confidence_markers if marker in lower_message)
            medium_count = sum(1 for marker in medium_confidence_markers if marker in lower_message)
            low_count = sum(1 for marker in low_confidence_markers if marker in lower_message)
            
            # Determine confidence based on most prevalent markers
            if high_count > medium_count and high_count > low_count:
                confidence = 0.9
            elif medium_count > low_count:
                confidence = 0.7
            elif low_count > 0:
                confidence = 0.4
            
        return confidence

    def share_knowledge(self, other_agent):
        """
        Share knowledge with another agent.
        
        Args:
            other_agent: Agent to share knowledge with
            
        Returns:
            Shared knowledge dictionary
        """
        # Implement shared knowledge - focus on task-relevant information
        shared_knowledge = {}
        
        # Share task understanding
        if "task_understanding" in self.knowledge_base:
            shared_knowledge["task_understanding"] = self.knowledge_base["task_understanding"]
            other_agent.add_to_knowledge_base("task_understanding", self.knowledge_base["task_understanding"])
        
        # Share domain knowledge
        #if "domain_knowledge" in self.knowledge_base:
            #shared_knowledge["domain_knowledge"] = self.knowledge_base["domain_knowledge"]
            #other_agent.add_to_knowledge_base("domain_knowledge", self.knowledge_base["domain_knowledge"])
        
        # Share reasoning approaches
        if "reasoning_approaches" in self.knowledge_base:
            shared_knowledge["reasoning_approaches"] = self.knowledge_base["reasoning_approaches"]
            other_agent.add_to_knowledge_base("reasoning_approaches", self.knowledge_base["reasoning_approaches"])
        
        self.logger.info(f"Agent {self.role} shared knowledge with {other_agent.role}")
        return shared_knowledge