"""
Mutual Trust implementation for agent system.

Mutual Trust: The shared belief that team members will perform their roles 
and protect the interests of their teammates.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple

from utils.prompts import TRUST_PROMPTS


class MutualTrust:
    """
    Implements mutual trust capabilities for agents.
    
    Mutual trust is the shared perception that team members will perform actions
    important to the team and will recognize and protect the interests of all team
    members engaged in their joint endeavor.
    """
    
    def __init__(self, initial_trust_factor: float = 0.8):
        """
        Initialize the mutual trust handler.
        
        Args:
            initial_trust_factor: Initial trust level between agents (0.0-1.0)
        """
        self.logger = logging.getLogger("teamwork.mutual_trust")
        self.logger.info(f"Initialized mutual trust handler with factor {initial_trust_factor}")
        
        # Set initial trust factor
        self.trust_factor = max(0.0, min(1.0, initial_trust_factor))
        
        # Track trust levels between agents
        self.trust_levels = {}  # {agent1: {agent2: trust_level}}
        
        # Track trust-related behaviors
        self.mistake_admissions = {}  # How often agents admit mistakes
        self.feedback_acceptance = {}  # How well agents accept feedback
        self.information_sharing_trust = {}  # Trust-influenced information sharing
    
    def initialize_trust_network(self, agent_roles: List[str]) -> None:
        """
        Initialize trust network between all agents.
        
        Args:
            agent_roles: List of agent roles
        """
        for role1 in agent_roles:
            if role1 not in self.trust_levels:
                self.trust_levels[role1] = {}
                
            for role2 in agent_roles:
                if role1 != role2:
                    # Apply some small random variation to initial trust
                    variation = random.uniform(-0.1, 0.1)
                    initial_trust = max(0.1, min(1.0, self.trust_factor + variation))
                    self.trust_levels[role1][role2] = initial_trust
        
        self.logger.info(f"Initialized trust network for {len(agent_roles)} agents")
    
    def get_trust_level(self, from_role: str, to_role: str) -> float:
        """
        Get trust level from one agent to another.
        
        Args:
            from_role: Role of the agent giving trust
            to_role: Role of the agent receiving trust
            
        Returns:
            Trust level (0.0-1.0)
        """
        if from_role not in self.trust_levels or to_role not in self.trust_levels[from_role]:
            # Default to base trust factor if not specified
            return self.trust_factor
            
        return self.trust_levels[from_role][to_role]
    
    def update_trust_level(self, from_role: str, to_role: str, event_type: str, magnitude: float = 0.05) -> None:
        """
        Update trust level based on events.
        
        Args:
            from_role: Role of the agent giving trust
            to_role: Role of the agent receiving trust
            event_type: Type of event affecting trust ('positive' or 'negative')
            magnitude: Magnitude of trust change
        """
        if from_role not in self.trust_levels:
            self.trust_levels[from_role] = {}
            
        if to_role not in self.trust_levels[from_role]:
            self.trust_levels[from_role][to_role] = self.trust_factor
            
        current_trust = self.trust_levels[from_role][to_role]
        
        if event_type == 'positive':
            # Increase trust (with diminishing returns as trust approaches 1.0)
            increase = magnitude * (1.0 - current_trust)
            new_trust = current_trust + increase
        else:  # negative
            # Decrease trust (with accelerating loss as trust approaches 0.0)
            decrease = magnitude * (1.0 / current_trust)
            new_trust = current_trust - decrease
            
        # Ensure trust stays in bounds
        new_trust = max(0.1, min(1.0, new_trust))
        self.trust_levels[from_role][to_role] = new_trust
        
        self.logger.info(f"Updated trust from {from_role} to {to_role}: {current_trust:.2f} -> {new_trust:.2f} ({event_type} event)")
    
    def track_mistake_admission(self, agent_role: str, admitted: bool) -> None:
        """
        Track whether an agent admitted a mistake.
        
        Args:
            agent_role: Role of the agent
            admitted: Whether the agent admitted a mistake
        """
        if agent_role not in self.mistake_admissions:
            self.mistake_admissions[agent_role] = {"admitted": 0, "total": 0}
            
        self.mistake_admissions[agent_role]["total"] += 1
        if admitted:
            self.mistake_admissions[agent_role]["admitted"] += 1
            
        self.logger.info(f"Agent {agent_role} {'admitted' if admitted else 'did not admit'} mistake")
    
    def track_feedback_acceptance(self, agent_role: str, feedback_from: str, accepted: bool) -> None:
        """
        Track whether an agent accepted feedback.
        
        Args:
            agent_role: Role of the agent receiving feedback
            feedback_from: Role of the agent giving feedback
            accepted: Whether the feedback was accepted
        """
        if agent_role not in self.feedback_acceptance:
            self.feedback_acceptance[agent_role] = {"accepted": 0, "total": 0}
            
        self.feedback_acceptance[agent_role]["total"] += 1
        if accepted:
            self.feedback_acceptance[agent_role]["accepted"] += 1
            
            # Positive trust event for accepting feedback
            self.update_trust_level(feedback_from, agent_role, 'positive', 0.03)
        else:
            # Negative trust event for rejecting feedback
            self.update_trust_level(feedback_from, agent_role, 'negative', 0.02)
            
        self.logger.info(f"Agent {agent_role} {'accepted' if accepted else 'rejected'} feedback from {feedback_from}")
    
    def evaluate_message_for_trust_indicators(self, message: str, sender_role: str, receiver_role: str) -> Dict[str, Any]:
        """
        Evaluate a message for trust indicators.
        
        Args:
            message: The message to evaluate
            sender_role: Role of the message sender
            receiver_role: Role of the message receiver
            
        Returns:
            Dictionary with trust indicators
        """
        indicators = {
            "admits_mistake": False,
            "accepts_feedback": False,
            "shares_information": False,
            "acknowledges_expertise": False,
            "expresses_vulnerability": False,
            "trust_score": 0.0
        }
        
        # Look for mistake admission
        mistake_indicators = [
            "i made a mistake", "i was wrong", "i misunderstood", 
            "i overlooked", "my error", "i missed", "i need to correct"
        ]
        
        indicators["admits_mistake"] = any(indicator in message.lower() for indicator in mistake_indicators)
        
        # Look for feedback acceptance
        feedback_indicators = [
            "good point", "i appreciate your feedback", "you're right", 
            "thank you for pointing that out", "that's helpful", 
            "i'll incorporate your suggestion"
        ]
        
        indicators["accepts_feedback"] = any(indicator in message.lower() for indicator in feedback_indicators)
        
        # Look for information sharing
        sharing_indicators = [
            "let me share", "i want to inform you", "here's what i know", 
            "from my perspective", "i found that", "i believe"
        ]
        
        indicators["shares_information"] = any(indicator in message.lower() for indicator in sharing_indicators)
        
        # Look for expertise acknowledgment
        expertise_indicators = [
            "given your expertise", "as you would know", "i trust your judgment", 
            "i value your perspective", "i defer to your knowledge", 
            "with your experience"
        ]
        
        indicators["acknowledges_expertise"] = any(indicator in message.lower() for indicator in expertise_indicators)
        
        # Look for vulnerability expressions
        vulnerability_indicators = [
            "i'm not sure", "i need help with", "i'm uncertain about", 
            "could you explain", "i'm having difficulty", "i don't understand"
        ]
        
        indicators["expresses_vulnerability"] = any(indicator in message.lower() for indicator in vulnerability_indicators)
        
        # Calculate overall score
        score_components = [
            indicators["admits_mistake"],
            indicators["accepts_feedback"],
            indicators["shares_information"],
            indicators["acknowledges_expertise"],
            indicators["expresses_vulnerability"]
        ]
        
        indicators["trust_score"] = sum(1 for component in score_components if component) / len(score_components)
        
        # Update trust levels based on indicators
        trust_impact = {
            "admits_mistake": 0.04,
            "accepts_feedback": 0.03,
            "shares_information": 0.02,
            "acknowledges_expertise": 0.03,
            "expresses_vulnerability": 0.04
        }
        
        for indicator, value in indicators.items():
            if value and indicator in trust_impact:
                self.update_trust_level(receiver_role, sender_role, 'positive', trust_impact[indicator])
        
        return indicators
    
    def filter_information_for_trust(self, information: str, trust_level: float) -> str:
        """
        Filter information based on trust level.
        
        Args:
            information: Full information to potentially share
            trust_level: Current trust level
            
        Returns:
            Filtered information appropriate for trust level
        """
        # At high trust levels, share full information
        if trust_level >= 0.8:
            return information
            
        # At medium trust levels, share most information
        elif trust_level >= 0.5:
            lines = information.split('\n')
            # Remove ~20% of the content
            filtered_lines = lines[:int(len(lines) * 0.8)]
            return '\n'.join(filtered_lines)
            
        # At low trust levels, share only basic information
        else:
            lines = information.split('\n')
            # Only share about half the information
            filtered_lines = lines[:max(1, int(len(lines) * 0.5))]
            return '\n'.join(filtered_lines) + "\n\n[Note: Limited information shared due to trust constraints]"
    
    def enhance_agent_prompt(self, base_prompt: str, agent_role: str, team_roles: List[str]) -> str:
        """
        Enhance an agent's prompt with mutual trust elements.
        
        Args:
            base_prompt: The original prompt for the agent
            agent_role: Role of this agent
            team_roles: Roles of all team members
            
        Returns:
            Enhanced prompt with mutual trust elements
        """
        # Calculate average trust levels
        trust_to_others = 0.0
        trust_from_others = 0.0
        trust_count = 0
        
        for other_role in team_roles:
            if other_role != agent_role:
                # Trust this agent gives to others
                if agent_role in self.trust_levels and other_role in self.trust_levels[agent_role]:
                    trust_to_others += self.trust_levels[agent_role][other_role]
                    trust_count += 1
                    
                # Trust others give to this agent
                if other_role in self.trust_levels and agent_role in self.trust_levels[other_role]:
                    trust_from_others += self.trust_levels[other_role][agent_role]
                    
        # Calculate averages
        avg_trust_to_others = trust_to_others / max(1, trust_count)
        avg_trust_from_others = trust_from_others / max(1, trust_count)
        
        # Adjust prompt based on trust levels
        if avg_trust_to_others >= 0.7 and avg_trust_from_others >= 0.7:
            trust_addition = TRUST_PROMPTS["high_trust_environment"]
        elif avg_trust_to_others >= 0.4 and avg_trust_from_others >= 0.4:
            trust_addition = TRUST_PROMPTS["medium_trust_environment"]
        else:
            trust_addition = TRUST_PROMPTS["low_trust_environment"]
        
        return base_prompt + trust_addition
    
    def get_trust_metrics(self) -> Dict[str, Any]:
        """
        Get metrics on mutual trust effectiveness.
        
        Returns:
            Dictionary with mutual trust metrics
        """
        metrics = {
            "average_trust_level": 0.0,
            "trust_network": self.trust_levels,
            "mistake_admission_rates": {},
            "feedback_acceptance_rates": {},
            "trust_environment": "medium"
        }
        
        # Calculate average trust level across all pairs
        total_trust = 0.0
        trust_count = 0
        
        for from_role, trust_dict in self.trust_levels.items():
            for to_role, trust_level in trust_dict.items():
                total_trust += trust_level
                trust_count += 1
                
        if trust_count > 0:
            metrics["average_trust_level"] = total_trust / trust_count
            
        # Calculate mistake admission rates
        for agent_role, data in self.mistake_admissions.items():
            if data["total"] > 0:
                rate = data["admitted"] / data["total"]
                metrics["mistake_admission_rates"][agent_role] = rate
                
        # Calculate feedback acceptance rates
        for agent_role, data in self.feedback_acceptance.items():
            if data["total"] > 0:
                rate = data["accepted"] / data["total"]
                metrics["feedback_acceptance_rates"][agent_role] = rate
                
        # Determine trust environment level
        avg_trust = metrics["average_trust_level"]
        if avg_trust >= 0.7:
            metrics["trust_environment"] = "high"
        elif avg_trust <= 0.4:
            metrics["trust_environment"] = "low"
            
        return metrics