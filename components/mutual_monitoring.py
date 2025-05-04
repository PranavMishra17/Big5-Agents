"""
Mutual Performance Monitoring implementation for agent system.

Mutual Performance Monitoring: The ability to develop common understandings
of the team environment and apply appropriate task strategies to accurately 
monitor teammate performance.
"""

import logging
from typing import Dict, List, Any

class MutualMonitoring:
    """
    Implements mutual performance monitoring capabilities for agents.
    
    Mutual performance monitoring enables agents to keep track of each other's work,
    identify potential errors or oversights, and provide corrective feedback.
    This becomes particularly important in stressful or complex tasks.
    """
    
    def __init__(self):
        """Initialize the mutual monitoring handler."""
        self.logger = logging.getLogger("teamwork.mutual_monitoring")
        self.performance_logs = {}
        self.feedback_history = []
        self.error_detections = []
        self.logger.info("Initialized mutual performance monitoring handler")
    
    def monitor_agent_response(self, agent_role: str, response: str, reasoning: str = None) -> Dict[str, Any]:
        """
        Monitor an agent's response for potential issues.
        
        Args:
            agent_role: Role of the agent being monitored
            response: The agent's direct response to the task
            reasoning: Optional reasoning provided by the agent
            
        Returns:
            Dictionary of monitoring results and potential issues
        """
        self.logger.info(f"Monitoring response from {agent_role}")
        
        # Store performance data
        self.performance_logs[agent_role] = {
            "response": response,
            "reasoning": reasoning or ""
        }
        
        # Check for potential issues
        issues = []
        
        # Get task type to determine appropriate checks
        import config
        task_type = config.TASK["type"]
        
        if task_type == "ranking":
            issues.extend(self._check_ranking_response(response))
        elif task_type == "mcq":
            issues.extend(self._check_mcq_response(response))
        else:
            issues.extend(self._check_general_response(response))
        
        # Check for reasoning quality issues
        if reasoning:
            issues.extend(self._check_reasoning_quality(reasoning))
        
        result = {
            "agent_role": agent_role,
            "issues_detected": len(issues) > 0,
            "issues": issues
        }
        
        # Log the monitoring results
        if issues:
            self.logger.info(f"Detected {len(issues)} issues in {agent_role}'s response")
            for issue in issues:
                self.logger.info(f"  - {issue['type']}: {issue['description']}")
        else:
            self.logger.info(f"No issues detected in {agent_role}'s response")
        
        return result
    
    def _check_ranking_response(self, response: str) -> List[Dict[str, Any]]:
        """Check for issues in a ranking response."""
        issues = []
        import config
        
        # Extract items mentioned in the response
        mentioned_items = []
        for item in config.TASK["options"]:
            if item.lower() in response.lower():
                mentioned_items.append(item)
        
        # Check for missing items
        missing_items = set(config.TASK["options"]) - set(mentioned_items)
        if missing_items:
            issues.append({
                "type": "missing_items",
                "description": f"Missing items in ranking: {', '.join(list(missing_items)[:3])}{'...' if len(missing_items) > 3 else ''}",
                "severity": "high"
            })
        
        # Check for potential duplicates
        for item in mentioned_items:
            # Count occurrences of the item in the response
            item_lower = item.lower()
            response_lower = response.lower()
            
            # Check for suspicious multiple occurrences
            # This is a simple heuristic and might need refinement
            start_idx = 0
            occurrences = 0
            
            while True:
                idx = response_lower.find(item_lower, start_idx)
                if idx == -1:
                    break
                
                occurrences += 1
                start_idx = idx + len(item_lower)
            
            if occurrences > 1:
                # This could be a duplicate ranking, but might also be legitimate repetition
                # So we mark it as a potential issue with medium severity
                issues.append({
                    "type": "potential_duplicate",
                    "description": f"Item '{item}' may be listed multiple times ({occurrences} occurrences)",
                    "severity": "medium"
                })
        
        # Check if any of the top 3 critical items are ranked very low (for NASA task)
        if "Lunar Survival" in config.TASK["name"]:
            critical_items = ["Oxygen tanks", "Water", "Stellar map"]
            for item in critical_items:
                # Try to find where this item is ranked
                for i in range(10, 16):  # Look for rankings 10-15
                    if f"{i}. {item}" in response or f"{i}. {item.lower()}" in response.lower():
                        issues.append({
                            "type": "questionable_ranking",
                            "description": f"Critical item '{item}' appears to be ranked very low at position {i}",
                            "severity": "medium"
                        })
                        break
        
        return issues
    
    def _check_mcq_response(self, response: str) -> List[Dict[str, Any]]:
        """Check for issues in a multiple-choice response."""
        issues = []
        import config
        
        # Check if a clear option selection is made
        clear_selection = False
        selected_option = None
        
        # Look for option identifiers (A, B, C, D, etc.)
        for option in config.TASK["options"]:
            option_id = option.split('.')[0].strip() if '.' in option else None
            if option_id:
                selection_patterns = [
                    f"Option {option_id}", 
                    f"select {option_id}", 
                    f"choose {option_id}",
                    f"answer is {option_id}",
                    f"answer: {option_id}"
                ]
                
                for pattern in selection_patterns:
                    if pattern.lower() in response.lower():
                        clear_selection = True
                        selected_option = option_id
                        break
                
                if clear_selection:
                    break
        
        if not clear_selection:
            issues.append({
                "type": "unclear_selection",
                "description": "No clear option selection (A, B, C, etc.) identified in the response",
                "severity": "high"
            })
        
        # Check for potential contradictions
        if clear_selection:
            # Look for indications of other options being selected
            for option in config.TASK["options"]:
                option_id = option.split('.')[0].strip() if '.' in option else None
                if option_id and option_id != selected_option:
                    selection_patterns = [
                        f"Option {option_id} is correct", 
                        f"select {option_id}", 
                        f"choose {option_id}",
                        f"answer is {option_id}",
                        f"answer: {option_id}"
                    ]
                    
                    for pattern in selection_patterns:
                        if pattern.lower() in response.lower():
                            issues.append({
                                "type": "contradictory_selection",
                                "description": f"Potential contradiction: response indicates {selected_option} but also mentions selecting {option_id}",
                                "severity": "high"
                            })
                            break
        
        return issues
    
    def _check_general_response(self, response: str) -> List[Dict[str, Any]]:
        """Check for issues in a general response."""
        issues = []
        import config
        
        # Check for completeness
        if len(response.split()) < 50:  # Very short response
            issues.append({
                "type": "incomplete_response",
                "description": "Response appears to be unusually short and may be incomplete",
                "severity": "medium"
            })
        
        # Check if all required elements are addressed (for tasks with evaluation criteria)
        if "evaluation_criteria" in config.TASK:
            for criterion in config.TASK["evaluation_criteria"]:
                # Create simple keyword check based on criterion
                keywords = [word.lower() for word in criterion.split() if len(word) > 4]
                if not any(keyword in response.lower() for keyword in keywords):
                    issues.append({
                        "type": "missing_criterion",
                        "description": f"Response may not address: '{criterion}'",
                        "severity": "medium"
                    })
        
        return issues
    
    def _check_reasoning_quality(self, reasoning: str) -> List[Dict[str, Any]]:
        """Check for issues in reasoning quality."""
        issues = []
        
        # Check for circular reasoning
        if reasoning.lower().count("because") < 2 and reasoning.lower().count("therefore") < 1 and reasoning.lower().count("hence") < 1:
            issues.append({
                "type": "limited_reasoning",
                "description": "Response may lack sufficient causal reasoning or justification",
                "severity": "medium"
            })
        
        # Check for inconsistencies in logic
        contradictory_pairs = [
            ("increase", "decrease"),
            ("more", "less"),
            ("higher", "lower"),
            ("best", "worst"),
            ("always", "never"),
            ("all", "none")
        ]
        
        for term1, term2 in contradictory_pairs:
            if term1 in reasoning.lower() and term2 in reasoning.lower():
                # This is a simple heuristic and might yield false positives
                issues.append({
                    "type": "potential_contradiction",
                    "description": f"Reasoning contains potentially contradictory terms: '{term1}' and '{term2}'",
                    "severity": "low"
                })
        
        # Look for overconfidence
        certainty_indicators = ["certainly", "definitely", "absolutely", "without doubt", "unquestionably"]
        if any(indicator in reasoning.lower() for indicator in certainty_indicators):
            issues.append({
                "type": "overconfidence",
                "description": "Reasoning shows signs of overconfidence which may indicate insufficient consideration of alternatives",
                "severity": "low"
            })
        
        return issues
    
    def generate_feedback(self, monitoring_result: Dict[str, Any], feedback_agent_role: str) -> str:
        """
        Generate feedback based on monitoring results.
        
        Args:
            monitoring_result: Results from monitoring an agent's response
            feedback_agent_role: Role of the agent providing feedback
            
        Returns:
            Feedback message to be conveyed to the monitored agent
        """
        agent_role = monitoring_result["agent_role"]
        issues = monitoring_result["issues"]
        
        if not issues:
            feedback = f"As your teammate, I've reviewed your response and don't see any major issues. Your approach seems sound, and I agree with your general assessment."
            self.logger.info(f"{feedback_agent_role} provided positive feedback to {agent_role}")
        else:
            feedback = f"As your teammate, I've reviewed your response and would like to offer some observations:\n\n"
            
            # Group issues by severity
            high_issues = [issue for issue in issues if issue["severity"] == "high"]
            medium_issues = [issue for issue in issues if issue["severity"] == "medium"]
            low_issues = [issue for issue in issues if issue["severity"] == "low"]
            
            # Start with high severity issues
            if high_issues:
                feedback += "Critical points to address:\n"
                for i, issue in enumerate(high_issues):
                    feedback += f"{i+1}. {issue['description']}\n"
                feedback += "\n"
            
            # Then medium severity
            if medium_issues:
                feedback += "Considerations to improve your response:\n"
                for i, issue in enumerate(medium_issues):
                    feedback += f"{i+1}. {issue['description']}\n"
                feedback += "\n"
            
            # Finally low severity
            if low_issues:
                feedback += "Minor points to consider:\n"
                for i, issue in enumerate(low_issues[:2]):  # Limit to top 2 low severity issues
                    feedback += f"{i+1}. {issue['description']}\n"
                feedback += "\n"
            
            feedback += "Please consider these points when refining your response. I appreciate your expertise and am looking forward to your perspective."
            self.logger.info(f"{feedback_agent_role} provided constructive feedback to {agent_role} on {len(issues)} issues")
        
        # Store feedback for later analysis
        self.feedback_history.append({
            "from_role": feedback_agent_role,
            "to_role": agent_role,
            "feedback": feedback,
            "issues": issues
        })
        
        return feedback
    
    def analyze_team_performance(self) -> Dict[str, Any]:
        """
        Analyze overall team performance in monitoring and feedback.
        
        Returns:
            Dictionary with team performance metrics
        """
        # Calculate metrics
        total_issues_detected = sum(len(feedback["issues"]) for feedback in self.feedback_history)
        total_feedback_exchanges = len(self.feedback_history)
        avg_issues_per_exchange = total_issues_detected / max(1, total_feedback_exchanges)
        
        # Analyze types of issues
        issue_types = {}
        for feedback in self.feedback_history:
            for issue in feedback["issues"]:
                issue_type = issue["type"]
                if issue_type not in issue_types:
                    issue_types[issue_type] = 0
                issue_types[issue_type] += 1
        
        # Analyze if feedback was addressed
        addressed_issues = 0
        for i in range(len(self.feedback_history) - 1):
            current_feedback = self.feedback_history[i]
            next_feedback = self.feedback_history[i + 1]
            
            if current_feedback["to_role"] == next_feedback["from_role"]:
                # Check if issues were addressed in next exchange
                current_issues = {issue["description"] for issue in current_feedback["issues"]}
                next_issues = {issue["description"] for issue in next_feedback["issues"]}
                addressed_issues += len(current_issues - next_issues)
        
        issue_resolution_rate = addressed_issues / max(1, total_issues_detected)
        
        # Prepare analysis
        analysis = {
            "total_monitoring_exchanges": total_feedback_exchanges,
            "total_issues_detected": total_issues_detected,
            "avg_issues_per_exchange": avg_issues_per_exchange,
            "issue_types": issue_types,
            "issue_resolution_rate": issue_resolution_rate,
            "team_monitoring_effectiveness": "high" if issue_resolution_rate > 0.7 else "medium" if issue_resolution_rate > 0.4 else "low"
        }
        
        self.logger.info(f"Team performance analysis completed: {analysis['team_monitoring_effectiveness']} effectiveness")
        return analysis
    
    def enhance_agent_prompt(self, base_prompt: str, monitoring_data: Dict[str, Any] = None) -> str:
        """
        Enhance an agent's prompt with mutual monitoring awareness.
        
        Args:
            base_prompt: The original prompt for the agent
            monitoring_data: Optional monitoring data from previous iterations
            
        Returns:
            Enhanced prompt with mutual monitoring elements
        """
        enhanced_prompt = base_prompt
        
        # Add mutual monitoring awareness to the prompt
        monitoring_addition = """
        
        As you develop your response, be aware that your teammate will be monitoring your work
        and may provide feedback. Similarly, you should monitor your teammate's work by:
        
        1. Checking for completeness and ensuring all required elements are addressed
        2. Identifying any potential inconsistencies in their reasoning
        3. Assessing if their conclusions align with the relevant principles for this task
        4. Considering if they've overlooked any critical factors
        
        If you notice issues, provide specific, constructive feedback that can help improve the team's overall response.
        """
        
        # Add specific insights from previous monitoring if available
        if monitoring_data and "feedback_history" in monitoring_data and monitoring_data["feedback_history"]:
            last_feedback = monitoring_data["feedback_history"][-1]
            if "to_role" in last_feedback and "feedback" in last_feedback:
                monitoring_addition += f"""
                
                In previous exchanges, the following feedback was provided:
                "{last_feedback['feedback']}"
                
                Consider this feedback as you develop your current response.
                """
        
        enhanced_prompt += monitoring_addition
        return enhanced_prompt