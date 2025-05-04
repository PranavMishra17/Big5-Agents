"""
Team Orientation implementation for agent system.

Team Orientation: The propensity to take others' behavior into account during 
group interaction and the belief in the importance of team goals over individual goals.
"""

import logging
from typing import Dict, List, Any, Optional

from utils.prompts import ORIENTATION_PROMPTS

class TeamOrientation:
    """
    Implements team orientation capabilities for agents.
    
    Team orientation is not just a preference for working with others but a tendency to enhance
    individual performance through the coordination, evaluation, and utilization of task inputs
    from team members while performing group tasks.
    """
    
    def __init__(self):
        """Initialize the team orientation handler."""
        self.logger = logging.getLogger("teamwork.team_orientation")
        self.logger.info("Initialized team orientation handler")
        
        # Track team orientation metrics
        self.input_consideration = {}  # How often an agent considers others' input
        self.information_sharing = {}  # How much info each agent shares
        self.goal_alignment = {}  # How aligned agent goals are with team goals
        
    def track_input_consideration(self, agent_role: str, considered_input: bool) -> None:
        """
        Track whether an agent considered input from teammates.
        
        Args:
            agent_role: Role of the agent
            considered_input: Whether the agent considered teammates' input
        """
        if agent_role not in self.input_consideration:
            self.input_consideration[agent_role] = {"considered": 0, "total": 0}
            
        self.input_consideration[agent_role]["total"] += 1
        if considered_input:
            self.input_consideration[agent_role]["considered"] += 1
            
        self.logger.info(f"Agent {agent_role} {'considered' if considered_input else 'ignored'} teammate input")
    
    def track_information_sharing(self, agent_role: str, shared_info: str) -> None:
        """
        Track information shared by an agent.
        
        Args:
            agent_role: Role of the agent
            shared_info: Information shared by the agent
        """
        if agent_role not in self.information_sharing:
            self.information_sharing[agent_role] = []
            
        self.information_sharing[agent_role].append(shared_info)
        self.logger.info(f"Agent {agent_role} shared information with the team")
    
    def evaluate_message_for_team_orientation(self, message: str) -> Dict[str, Any]:
        """
        Evaluate a message for signs of team orientation.
        
        Args:
            message: The message to evaluate
            
        Returns:
            Dictionary with team orientation indicators
        """
        indicators = {
            "considers_team_input": False,
            "shares_information": False,
            "prioritizes_team_goals": False,
            "participatory_goal_setting": False,
            "strategic_thinking": False,
            "team_orientation_score": 0.0
        }
        
        # Look for indicators of considering others' input
        input_indicators = [
            "considering your point", "you raise a good point", 
            "as you suggested", "taking into account your", 
            "building on what you said", "agree with your assessment"
        ]
        
        indicators["considers_team_input"] = any(indicator in message.lower() for indicator in input_indicators)
        
        # Look for information sharing
        sharing_indicators = [
            "i want to share", "let me provide", "here's some information",
            "i found that", "according to my expertise", "from my perspective"
        ]
        
        indicators["shares_information"] = any(indicator in message.lower() for indicator in sharing_indicators)
        
        # Look for team goal prioritization
        goal_indicators = [
            "our goal", "team objective", "we should focus on", 
            "our priority", "best for the team", "achieve together"
        ]
        
        indicators["prioritizes_team_goals"] = any(indicator in message.lower() for indicator in goal_indicators)
        
        # Look for participatory goal setting
        goal_setting_indicators = [
            "we should aim to", "i suggest we focus on", 
            "our approach should be", "we could work toward", 
            "let's establish", "our strategy should"
        ]
        
        indicators["participatory_goal_setting"] = any(indicator in message.lower() for indicator in goal_setting_indicators)
        
        # Look for strategic thinking
        strategy_indicators = [
            "strategic approach", "methodical way", "systematic process", 
            "structured analysis", "step-by-step", "break down the problem"
        ]
        
        indicators["strategic_thinking"] = any(indicator in message.lower() for indicator in strategy_indicators)
        
        # Calculate overall score
        score_components = [
            indicators["considers_team_input"],
            indicators["shares_information"],
            indicators["prioritizes_team_goals"],
            indicators["participatory_goal_setting"],
            indicators["strategic_thinking"]
        ]
        
        indicators["team_orientation_score"] = sum(1 for component in score_components if component) / len(score_components)
        
        return indicators
    
    def enhance_agent_prompt(self, base_prompt: str) -> str:
        """Enhance an agent's prompt with team orientation elements."""
        team_orientation_addition = ORIENTATION_PROMPTS["enhanced_orientation"]
        return base_prompt + team_orientation_addition

    def get_team_orientation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics on team orientation effectiveness.
        
        Returns:
            Dictionary with team orientation metrics
        """
        metrics = {
            "input_consideration": {},
            "information_sharing": {},
            "overall_team_orientation": "medium"
        }
        
        # Calculate input consideration rates
        for agent_role, data in self.input_consideration.items():
            if data["total"] > 0:
                rate = data["considered"] / data["total"]
                metrics["input_consideration"][agent_role] = rate
        
        # Calculate information sharing metrics
        for agent_role, info_list in self.information_sharing.items():
            metrics["information_sharing"][agent_role] = len(info_list)
        
        # Calculate overall team orientation level
        consideration_scores = list(metrics["input_consideration"].values())
        sharing_scores = list(metrics["information_sharing"].values())
        
        if consideration_scores and sum(consideration_scores) / len(consideration_scores) > 0.7:
            if sharing_scores and sum(sharing_scores) / len(sharing_scores) > 3:
                metrics["overall_team_orientation"] = "high"
        elif consideration_scores and sum(consideration_scores) / len(consideration_scores) < 0.3:
            if sharing_scores and sum(sharing_scores) / len(sharing_scores) < 1:
                metrics["overall_team_orientation"] = "low"
        
        return metrics