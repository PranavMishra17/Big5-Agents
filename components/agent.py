"""
Base agent class for modular agent system.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

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


class Agent:
    """Base agent class for modular agent system."""
    
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
                 examples: Optional[List[Dict[str, str]]] = None):
        """
        Initialize an LLM-based agent with a specific role.
        
        Args:
            role: The role of the agent (e.g., "Critical Analyst")
            expertise_description: Description of the agent's expertise
            use_team_leadership: Whether this agent uses team leadership behaviors
            use_closed_loop_comm: Whether this agent uses closed-loop communication
            use_mutual_monitoring: Whether this agent uses mutual performance monitoring
            use_shared_mental_model: Whether this agent uses shared mental models
            examples: Optional examples to include in the prompt
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
        
        # Initialize logger
        self.logger = logging.getLogger(f"agent.{role}")


        # Initialize LLM
        self.client = AzureChatOpenAI(
            azure_deployment=config.AZURE_DEPLOYMENT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT,
            temperature=config.TEMPERATURE
        )
        
        # Build initial system message
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
                
        self.logger.info(f"Initialized {self.role} agent")
    

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        # Base prompt
        prompt = AGENT_SYSTEM_PROMPTS["base"].format(
            role=self.role,
            expertise_description=self.expertise_description,
            team_name=config.TEAM_NAME,
            team_goal=config.TEAM_GOAL,
            task_name=config.TASK['name'],
            task_description=config.TASK['description'],
            task_type=config.TASK['type'],
            expected_output_format=config.TASK.get('expected_output_format', 'not specified')
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
    


    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: The message to send to the agent
            
        Returns:
            The agent's response
        """
        self.logger.info(f"Received message: {message[:100]}...")
        
        # Add the user message to the conversation
        self.messages.append({"role": "user", "content": message})
        
        # Get response from LLM - fix for LangChain API change
        try:
            # New LangChain API style
            response = self.client.invoke(self.messages)
            assistant_message = response.content
        except TypeError as e:
            if "missing 1 required positional argument: 'input'" in str(e):
                # Fall back to old style invocation
                self.logger.info("Falling back to older LangChain API style")
                response = self.client.invoke(input=self.messages)
                assistant_message = response.content
            else:
                # Re-raise if it's a different TypeError
                raise
        
        # Extract and store the response
        self.messages.append({"role": "assistant", "content": assistant_message})
        self.conversation_history.append({"user": message, "assistant": assistant_message})
        
        self.logger.info(f"Responded: {assistant_message[:100]}...")
        
        return assistant_message


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
        Extract the agent's response to the task from their message.
        
        Args:
            message: Optional message to analyze, defaults to last response
            
        Returns:
            Structured response based on task type
        """
        if message is None:
            if not self.conversation_history:
                return {}
            message = self.conversation_history[-1]["assistant"]
        
        task_type = config.TASK["type"]
        
        if task_type == "ranking":
            return self.extract_ranking(message)
        elif task_type == "mcq":
            return self.extract_mcq_answer(message)
        elif task_type == "multi_choice_mcq":
            return self.extract_multi_choice_mcq_answer(message)
        elif task_type == "yes_no_maybe":
            return self.extract_yes_no_maybe_answer(message)
        elif task_type in ["open_ended", "estimation", "selection"]:
            return {"response": message, "confidence": self.extract_confidence(message)}
        else:
            return {"raw_response": message}

    def extract_ranking(self, message):
        """
        Extract a ranking from the agent's response.
        
        Args:
            message: Message to analyze
            
        Returns:
            List of ranked items and confidence level
        """
        ranking = []
        lines = message.split('\n')
        
        # Look for numbered items (1. Item, 2. Item, etc.)
        for line in lines:
            for i in range(1, len(config.TASK["options"]) + 1):
                if f"{i}." in line or f"{i}:" in line:
                    for item in config.TASK["options"]:
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
        for item in config.TASK["options"]:
            if item not in seen_items:
                valid_ranking.append(item)
                seen_items.add(item)
        
        # Extract confidence level
        confidence = self.extract_confidence(message)
        
        return {
            "ranking": valid_ranking[:len(config.TASK["options"])],
            "confidence": confidence
        }
    
    def extract_mcq_answer(self, message):
        """
        Extract an MCQ answer from the agent's response.
        
        Args:
            message: Message to analyze
            
        Returns:
            MCQ answer and confidence level
        """
        # Look for option identifiers (A, B, C, D, etc.)
        for line in message.split('\n'):
            line = line.strip()
            for option in config.TASK["options"]:
                option_id = option.split('.')[0].strip() if '.' in option else None
                if option_id and (line.startswith(option_id) or f"Option {option_id}" in line or f"Answer: {option_id}" in line):
                    # Extract confidence level
                    confidence = self.extract_confidence(message)
                    return {"answer": option_id, "confidence": confidence}
        
        # Extract confidence level
        confidence = self.extract_confidence(message)
        
        # If no explicit option identifier is found, look for the full option text
        for line in message.split('\n'):
            for option in config.TASK["options"]:
                # Extract option identifier (A, B, C, etc.)
                option_id = option.split('.')[0].strip() if '.' in option else None
                # Check if the full option text is in the line
                if option[2:].strip().lower() in line.lower():
                    return {"answer": option_id, "confidence": confidence}
        
        # If still not found, try to determine if there's any indication of an answer
        lower_message = message.lower()
        for option in config.TASK["options"]:
            option_id = option.split('.')[0].strip() if '.' in option else None
            if option_id and f"select {option_id.lower()}" in lower_message or f"choose {option_id.lower()}" in lower_message:
                return {"answer": option_id, "confidence": confidence}
        
        # No clear answer found
        return {"answer": None, "confidence": confidence}
    
    def extract_yes_no_maybe_answer(self, message):
        """
        Extract a yes/no/maybe answer from the agent's response.
        
        Args:
            message: Message to analyze
            
        Returns:
            Yes/no/maybe answer and confidence level
        """
        import re
        
        # Look for explicit answer patterns
        answer_patterns = [
            r"ANSWER:\s*(yes|no|maybe)",
            r"FINAL ANSWER:\s*(yes|no|maybe)",
            r"answer is:\s*(yes|no|maybe)",
            r"my answer:\s*(yes|no|maybe)",
            r"the answer is:\s*(yes|no|maybe)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, message.lower(), re.IGNORECASE)
            if match:
                answer = match.group(1).lower()
                confidence = self.extract_confidence(message)
                return {"answer": answer, "confidence": confidence}
        
        # Look for clear yes/no/maybe statements in the text
        lower_message = message.lower()
        
        # Check for definitive yes
        if any(phrase in lower_message for phrase in ["yes,", "yes.", "yes\n", "answer: yes", "answer is yes"]):
            confidence = self.extract_confidence(message)
            return {"answer": "yes", "confidence": confidence}
        
        # Check for definitive no
        if any(phrase in lower_message for phrase in ["no,", "no.", "no\n", "answer: no", "answer is no"]):
            confidence = self.extract_confidence(message)
            return {"answer": "no", "confidence": confidence}
        
        # Check for maybe/uncertain
        if any(phrase in lower_message for phrase in ["maybe", "uncertain", "unclear", "insufficient evidence", "cannot determine"]):
            confidence = self.extract_confidence(message)
            return {"answer": "maybe", "confidence": confidence}
        
        # No clear answer found
        confidence = self.extract_confidence(message)
        return {"answer": None, "confidence": confidence}
    
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
        if "domain_knowledge" in self.knowledge_base:
            shared_knowledge["domain_knowledge"] = self.knowledge_base["domain_knowledge"]
            other_agent.add_to_knowledge_base("domain_knowledge", self.knowledge_base["domain_knowledge"])
        
        # Share reasoning approaches
        if "reasoning_approaches" in self.knowledge_base:
            shared_knowledge["reasoning_approaches"] = self.knowledge_base["reasoning_approaches"]
            other_agent.add_to_knowledge_base("reasoning_approaches", self.knowledge_base["reasoning_approaches"])
        
        self.logger.info(f"Agent {self.role} shared knowledge with {other_agent.role}")
        return shared_knowledge