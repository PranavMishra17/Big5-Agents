"""
Shared Mental Model implementation for agent system.

Shared Mental Model: An organizing knowledge structure of the relationships
among the task the team is engaged in and how the team members will interact.
This enables team members to anticipate and predict each other's needs through
common understandings of the environment and expectations of performance.
"""

import logging
import json
from typing import Dict, List, Any

class SharedMentalModel:
    """
    Implements shared mental model capabilities for agents.
    
    A shared mental model enables agents to have common understanding of:
    1. Task-related knowledge (task procedures, strategies, environment constraints)
    2. Team-related knowledge (roles, responsibilities, interaction patterns)
    
    This shared understanding helps with coordination and anticipation of needs.
    """
    
    def __init__(self):
        """Initialize the shared mental model handler."""
        self.logger = logging.getLogger("teamwork.shared_mental_model")
        
        # Initialize knowledge repositories
        self.task_knowledge = {}
        self.team_knowledge = {}
        self.shared_understanding = {}
        self.convergence_metrics = []
        
        self.logger.info("Initialized shared mental model handler")
    
    def initialize_task_model(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the shared task-related mental model.
        
        Args:
            task_info: Task information from config
            
        Returns:
            Dictionary with task model elements
        """
        # Create task-related mental model
        task_type = task_info.get("type", "general")
        
        if task_type == "ranking":
            task_model = self._initialize_ranking_task_model(task_info)
        elif task_type == "mcq":
            task_model = self._initialize_mcq_task_model(task_info)
        else:
            task_model = self._initialize_general_task_model(task_info)
        
        # Store in task knowledge repository
        self.task_knowledge["current_task"] = task_model
        
        self.logger.info(f"Initialized task mental model for {task_info['name']}")
        return task_model
    
    def _initialize_ranking_task_model(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize mental model for a ranking task."""
        return {
            "objective": f"Rank {len(task_info.get('options', []))} items in order of importance",
            "task_type": "ranking",
            "items_to_rank": task_info.get("options", []),
            "evaluation_criteria": {
                "completeness": "All items must be included in the ranking",
                "consistency": "Ranking should be internally consistent in its logic",
                "justification": "Each item's position should have clear reasoning",
                "domain_validity": "Ranking should align with relevant domain principles"
            },
            "task_procedure": [
                "Individual assessment of items based on specialized knowledge",
                "Sharing of perspectives and justifications",
                "Identification of areas of agreement and disagreement",
                "Resolution of disagreements through evidence-based discussion",
                "Creation of consensus ranking"
            ],
            "expected_output": task_info.get("expected_output_format", "Numbered list from 1 to n")
        }
    
    def _initialize_mcq_task_model(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize mental model for a multiple-choice task."""
        return {
            "objective": "Select the best answer from multiple options",
            "task_type": "mcq",
            "options": task_info.get("options", []),
            "evaluation_criteria": {
                "option_analysis": "Thorough analysis of each available option",
                "evidence_evaluation": "Consideration of relevant evidence for each option",
                "clear_selection": "Explicit selection of a single best answer",
                "reasoning_quality": "Strong justification for why the selected option is best"
            },
            "task_procedure": [
                "Individual analysis of each option by each team member",
                "Comparison of perspectives on option strengths and weaknesses",
                "Evaluation of evidence supporting different options",
                "Collaborative selection of best answer through reasoned discussion"
            ],
            "expected_output": task_info.get("expected_output_format", "Selected option with justification")
        }
    
    def _initialize_general_task_model(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize mental model for a general task."""
        return {
            "objective": task_info.get("description", "Complete the assigned task"),
            "task_type": task_info.get("type", "general"),
            "evaluation_criteria": task_info.get("evaluation_criteria", {
                "comprehensiveness": "Address all aspects of the task",
                "reasoning_quality": "Clear and sound reasoning process",
                "evidence_use": "Appropriate use of relevant evidence",
                "clarity": "Clear communication of conclusions"
            }),
            "task_procedure": [
                "Problem analysis and decomposition",
                "Individual contribution of specialized knowledge",
                "Synthesis of diverse perspectives",
                "Development of integrated solution",
                "Review and refinement"
            ],
            "expected_output": task_info.get("expected_output_format", "Comprehensive response to the task")
        }
    
    def initialize_team_model(self, roles: List[str]) -> Dict[str, Any]:
        """
        Initialize the shared team-related mental model.
        
        Args:
            roles: List of agent roles in the team
            
        Returns:
            Dictionary with team model elements
        """
        # Create team-related mental model
        team_model = {
            "roles": {},
            "interaction_patterns": {
                "collaborative_problem_solving": "Joint work on identifying and addressing key issues",
                "knowledge_sharing": "Explicit communication of relevant domain knowledge",
                "perspective_integration": "Combining diverse viewpoints into unified understanding",
                "disagreement_resolution": "Evidence-based approach to resolving different perspectives"
            },
            "coordination_mechanisms": {
                "explicit_communication": "Clear expression of reasoning and justifications",
                "knowledge_integration": "Combining specialized knowledge for better decisions",
                "progress_tracking": "Monitoring advancement toward solution development"
            }
        }
        
        # Add role-specific information
        import config
        for role in roles:
            if role in config.AGENT_ROLES:
                role_expertise = config.AGENT_ROLES[role]
                
                # Extract key responsibilities and information needs based on role
                if "Critical Analyst" in role:
                    team_model["roles"][role] = {
                        "expertise": role_expertise,
                        "responsibilities": [
                            "Evaluate logical soundness of arguments",
                            "Identify potential biases or flaws in reasoning",
                            "Assess quality and sufficiency of evidence",
                            "Test robustness of conclusions"
                        ],
                        "information_needs": [
                            "Domain-specific principles and constraints",
                            "Context information for proper analysis",
                            "Diverse perspectives to analyze"
                        ]
                    }
                elif "Domain Expert" in role:
                    team_model["roles"][role] = {
                        "expertise": role_expertise,
                        "responsibilities": [
                            "Provide accurate domain knowledge",
                            "Apply relevant principles to the task",
                            "Identify domain-specific constraints and considerations",
                            "Ensure technical accuracy"
                        ],
                        "information_needs": [
                            "Clear task definition and constraints",
                            "Specific questions about domain applications",
                            "Areas where domain knowledge is most needed"
                        ]
                    }
                elif "Creative Strategist" in role:
                    team_model["roles"][role] = {
                        "expertise": role_expertise,
                        "responsibilities": [
                            "Generate innovative approaches",
                            "Identify non-obvious connections",
                            "Challenge conventional thinking",
                            "Envision alternative solutions"
                        ],
                        "information_needs": [
                            "Domain constraints to work within",
                            "Core problem definition",
                            "Critical factors for solution viability"
                        ]
                    }
                elif "Process Facilitator" in role:
                    team_model["roles"][role] = {
                        "expertise": role_expertise,
                        "responsibilities": [
                            "Ensure methodical evaluation process",
                            "Facilitate effective knowledge integration",
                            "Keep discussion focused on key objectives",
                            "Track progress toward solution"
                        ],
                        "information_needs": [
                            "Clear understanding of each team member's contributions",
                            "Areas of agreement and disagreement",
                            "Decision criteria and constraints"
                        ]
                    }
                else:
                    # Generic role information
                    team_model["roles"][role] = {
                        "expertise": role_expertise,
                        "responsibilities": [
                            "Contribute specialized knowledge",
                            "Evaluate information critically",
                            "Collaborate effectively with team members",
                            "Help develop integrated solution"
                        ],
                        "information_needs": [
                            "Clear task requirements",
                            "Other team members' perspectives",
                            "Areas where specialized input is needed"
                        ]
                    }
        
        # Store in team knowledge repository
        self.team_knowledge["current_team"] = team_model
        
        self.logger.info(f"Initialized team mental model with {len(roles)} roles")
        return team_model
    
    def update_shared_understanding(self, agent_role: str, understanding: Dict[str, Any]) -> None:
        """
        Update an agent's understanding in the shared mental model.
        
        Args:
            agent_role: Role of the agent sharing understanding
            understanding: Dictionary of the agent's current understanding
        """
        self.shared_understanding[agent_role] = understanding
        self.logger.info(f"Updated shared understanding from {agent_role}")
        
        # Calculate convergence metrics if multiple agents have shared
        if len(self.shared_understanding) > 1:
            convergence = self._calculate_understanding_convergence()
            self.convergence_metrics.append(convergence)
            self.logger.info(f"Current understanding convergence: {convergence['overall_convergence']:.2f}")
    
    def _calculate_understanding_convergence(self) -> Dict[str, Any]:
        """
        Calculate how well team members' mental models have converged.
        
        Returns:
            Dictionary with convergence metrics
        """
        # This is a simplified calculation - in a real implementation, 
        # this would use more sophisticated comparison algorithms
        convergence = {
            "timestamp": "current_time",
            "elements": {},
            "overall_convergence": 0.0
        }
        
        # Extract all agents
        agents = list(self.shared_understanding.keys())
        
        # Compare each pair of agents
        total_comparisons = 0
        total_similarity = 0.0
        
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                agent1 = agents[i]
                agent2 = agents[j]
                
                # Compare understandings
                understanding1 = self.shared_understanding[agent1]
                understanding2 = self.shared_understanding[agent2]
                
                # Check if both have key components to compare
                if "task_objective" in understanding1 and "task_objective" in understanding2:
                    similarity = self._calculate_text_similarity(
                        understanding1["task_objective"], 
                        understanding2["task_objective"]
                    )
                    convergence["elements"][f"task_objective_{agent1}_{agent2}"] = similarity
                    
                    total_similarity += similarity
                    total_comparisons += 1
                
                # Check if both have key factors
                if "key_factors" in understanding1 and "key_factors" in understanding2:
                    similarity = self._calculate_list_similarity(
                        understanding1["key_factors"], 
                        understanding2["key_factors"]
                    )
                    convergence["elements"][f"key_factors_{agent1}_{agent2}"] = similarity
                    
                    total_similarity += similarity
                    total_comparisons += 1
                
                # Check if both have approach understanding
                if "approach" in understanding1 and "approach" in understanding2:
                    similarity = self._calculate_text_similarity(
                        understanding1["approach"], 
                        understanding2["approach"]
                    )
                    convergence["elements"][f"approach_{agent1}_{agent2}"] = similarity
                    
                    total_similarity += similarity
                    total_comparisons += 1
        
        # Calculate overall convergence
        if total_comparisons > 0:
            convergence["overall_convergence"] = total_similarity / total_comparisons
        
        return convergence
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two text strings."""
        # This is a very simple similarity measure for illustration
        # In a real system, you'd use more sophisticated NLP techniques
        
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Create word sets
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / max(1, len(union))
        return similarity
    
    def _calculate_list_similarity(self, list1: List, list2: List) -> float:
        """Calculate similarity between two lists."""
        if not list1 or not list2:
            return 0.0
        
        # Check overlap
        set1 = set(str(item).lower() for item in list1)
        set2 = set(str(item).lower() for item in list2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Jaccard similarity
        similarity = len(intersection) / max(1, len(union))
        
        return similarity
    
    def enhance_agent_prompt(self, agent_role: str, base_prompt: str) -> str:
        """
        Enhance an agent's prompt with shared mental model elements.
        
        Args:
            agent_role: Role of the agent
            base_prompt: The original prompt for the agent
            
        Returns:
            Enhanced prompt with shared mental model elements
        """
        enhanced_prompt = base_prompt
        
        # Add shared task model
        if "current_task" in self.task_knowledge:
            task_model = self.task_knowledge["current_task"]
            
            # Add shared understanding of the task objective
            task_info = f"\n\nOur team's shared understanding of the task:\n"
            task_info += f"Objective: {task_model.get('objective', 'Not specified')}\n"
            
            # Add evaluation criteria
            criteria_info = "Key evaluation criteria:\n"
            for aspect, description in task_model.get("evaluation_criteria", {}).items():
                criteria_info += f"- {aspect.capitalize()}: {description}\n"
            
            # Add task procedure
            procedure_info = "\nOur agreed approach to this task:\n"
            for i, step in enumerate(task_model.get("task_procedure", [])):
                procedure_info += f"{i+1}. {step}\n"
            
            # Add expected output
            output_info = f"\nExpected output format: {task_model.get('expected_output', 'Not specified')}\n"
            
            # Combine shared mental model elements
            shared_model_info = f"{task_info}\n{criteria_info}\n{procedure_info}\n{output_info}"
            
            # Add role-specific guidance
            if "current_team" in self.team_knowledge and agent_role in self.team_knowledge["current_team"]["roles"]:
                role_info = self.team_knowledge["current_team"]["roles"][agent_role]
                
                role_guidance = f"\nAs the {agent_role}, your specific expertise is valued for this task.\n"
                role_guidance += "Your key responsibilities include:\n"
                for resp in role_info.get("responsibilities", []):
                    role_guidance += f"- {resp}\n"
                
                role_guidance += "\nYou should seek from your teammates:\n"
                for need in role_info.get("information_needs", []):
                    role_guidance += f"- {need}\n"
                
                shared_model_info += f"\n{role_guidance}"
            
            # Add convergence information if available
            if self.convergence_metrics:
                latest_convergence = self.convergence_metrics[-1]
                convergence_info = f"\nOur team's mental models have converged to a level of {latest_convergence['overall_convergence']:.2f} out of 1.0."
                shared_model_info += convergence_info
            
            # Add the shared mental model information to the prompt
            enhanced_prompt += f"\n\n=== SHARED TEAM UNDERSTANDING ===\n{shared_model_info}"
        
        return enhanced_prompt
    
    def extract_understanding_from_message(self, message: str) -> Dict[str, Any]:
        """
        Extract an agent's understanding from their message.
        
        Args:
            message: The agent's message
            
        Returns:
            Dictionary representing the agent's mental model
        """
        # This would ideally use NLP techniques to extract understanding
        # This is a simplified implementation
        understanding = {
            "task_objective": "",
            "key_factors": [],
            "approach": "",
            "terminology": {}
        }
        
        # Extract task objective
        objective_markers = [
            "our objective is", "the goal is", "we need to", "our task is",
            "we are trying to", "we aim to", "the purpose is"
        ]
        
        for marker in objective_markers:
            if marker in message.lower():
                parts = message.lower().split(marker, 1)
                if len(parts) > 1:
                    # Extract the sentence containing the objective
                    objective_part = parts[1].strip()
                    end_markers = [". ", ".\n", "\n\n"]
                    for end_marker in end_markers:
                        if end_marker in objective_part:
                            objective_part = objective_part.split(end_marker, 1)[0]
                            break
                    
                    understanding["task_objective"] = objective_part
                    break
        
        # Extract key factors
        key_factor_markers = [
            "key factors", "important considerations", "critical elements",
            "main points", "key aspects", "important factors"
        ]
        
        for marker in key_factor_markers:
            if marker in message.lower():
                parts = message.lower().split(marker, 1)
                if len(parts) > 1:
                    factors_part = parts[1].strip()
                    # Look for a list structure (numbered or bulleted)
                    lines = factors_part.split("\n")
                    factors = []
                    
                    for line in lines[:5]:  # Limit to first 5 lines after the marker
                        line = line.strip()
                        if line.startswith("- ") or line.startswith("* ") or any(f"{i}." in line for i in range(1, 6)):
                            # Remove list markers
                            factor = line.replace("- ", "").replace("* ", "")
                            for i in range(1, 6):
                                factor = factor.replace(f"{i}. ", "").replace(f"{i}.", "")
                            
                            factors.append(factor.strip())
                    
                    understanding["key_factors"] = factors
                    break
        
        # Extract approach understanding
        approach_markers = [
            "my approach", "our approach", "how we should tackle",
            "I suggest", "we should", "recommended approach"
        ]
        
        for marker in approach_markers:
            if marker in message.lower():
                parts = message.lower().split(marker, 1)
                if len(parts) > 1:
                    approach_part = parts[1].strip()
                    end_markers = ["\n\n", "in conclusion", "to summarize"]
                    for end_marker in end_markers:
                        if end_marker in approach_part:
                            approach_part = approach_part.split(end_marker, 1)[0]
                            break
                    
                    understanding["approach"] = approach_part
                    break
        
        # Extract terminology
        import re
        definition_patterns = [
            r'(\w+) refers to (.*?)\.',
            r'(\w+) is defined as (.*?)\.',
            r'(\w+) means (.*?)\.',
            r'by (\w+) I mean (.*?)\.'
        ]
        
        for pattern in definition_patterns:
            definitions = re.findall(pattern, message.lower())
            for term, definition in definitions:
                understanding["terminology"][term] = definition
        
        return understanding
    
    def analyze_mental_model_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of the shared mental model.
        
        Returns:
            Dictionary with effectiveness metrics
        """
        # Analyze convergence trend
        convergence_trend = "unknown"
        if len(self.convergence_metrics) > 1:
            initial = self.convergence_metrics[0]["overall_convergence"]
            final = self.convergence_metrics[-1]["overall_convergence"]
            
            if final > initial + 0.2:
                convergence_trend = "strong_improvement"
            elif final > initial + 0.1:
                convergence_trend = "moderate_improvement"
            elif final > initial:
                convergence_trend = "slight_improvement"
            elif final < initial - 0.1:
                convergence_trend = "divergence"
            else:
                convergence_trend = "stable"
        
        # Analyze understanding completeness
        completeness_scores = {}
        for role, understanding in self.shared_understanding.items():
            # Calculate completeness based on how many elements are present
            task_objective_completeness = 1.0 if understanding.get("task_objective") else 0.0
            key_factors_completeness = min(1.0, len(understanding.get("key_factors", [])) / 3.0)  # Assume 3+ factors is complete
            approach_completeness = 1.0 if understanding.get("approach") else 0.0
            terminology_completeness = min(1.0, len(understanding.get("terminology", {})) / 3.0)  # Assume 3+ terms is complete
            
            avg_completeness = (task_objective_completeness + key_factors_completeness + 
                              approach_completeness + terminology_completeness) / 4.0
            completeness_scores[role] = avg_completeness
        
        # Overall metrics
        analysis = {
            "convergence_trend": convergence_trend,
            "final_convergence": self.convergence_metrics[-1]["overall_convergence"] if self.convergence_metrics else 0.0,
            "understanding_completeness": completeness_scores,
            "avg_completeness": sum(completeness_scores.values()) / max(1, len(completeness_scores)),
            "effectiveness_rating": "high" if (self.convergence_metrics and self.convergence_metrics[-1]["overall_convergence"] > 0.7) else "medium" if (self.convergence_metrics and self.convergence_metrics[-1]["overall_convergence"] > 0.4) else "low"
        }
        
        self.logger.info(f"Mental model effectiveness analysis: {analysis['effectiveness_rating']} effectiveness")
        return analysis