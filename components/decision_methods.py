"""
Decision methods for aggregating agent responses.
"""

import logging
import re
from typing import Dict, List, Any, Tuple
import math
import config

class DecisionMethods:
    """
    Implements different decision methods for aggregating agent responses.
    Updated to support isolated task configuration.
    """
    
    def __init__(self, task_config: Dict[str, Any] = None):
        """Initialize with optional task configuration."""
        self.logger = logging.getLogger("decision.methods")
        self.task_config = task_config
        self.logger.info("Initialized decision methods handler")

    def set_task_config(self, task_config: Dict[str, Any]):
        """Set the task configuration for this decision session."""
        self.task_config = task_config


    def majority_voting(self, agent_responses, task_config: Dict[str, Any] = None):
        """Apply majority voting supporting A-J options."""
        task_config = task_config or self.task_config
        if not task_config:
            import config
            task_config = config.TASK
            
        task_type = task_config.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            return self._majority_voting_yes_no_maybe(agent_responses)
        
        votes = {}
        max_votes = 0
        
        # Get valid option letters for this task
        valid_options = self._get_option_letters(task_config)
        
        for agent_role, response in agent_responses.items():
            preference = None
            
            if isinstance(response, dict):
                if "final_decision" in response:
                    preference = self.extract_answer_option(response["final_decision"])
                elif "extract" in response and isinstance(response["extract"], dict):
                    if "answer" in response["extract"]:
                        preference = response["extract"]["answer"]
                        if preference:
                            preference = preference.upper()
            elif isinstance(response, str):
                preference = self.extract_answer_option(response)
            
            # Register vote if valid option
            if preference and preference in valid_options:
                if preference not in votes:
                    votes[preference] = 0
                votes[preference] += 1
                max_votes = max(max_votes, votes[preference])
                logging.info(f"Recorded vote from {agent_role}: {preference}")
        
        winning_option = None if not votes else max(votes, key=votes.get)
        total_votes = sum(votes.values())
        
        return {
            "method": "majority_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "total_votes": total_votes,
            "confidence": max_votes / total_votes if total_votes > 0 else 0
        }

    def borda_count(self, agent_responses, task_config: Dict[str, Any] = None):
        """Apply Borda count supporting A-J options."""
        task_config = task_config or self.task_config
        if not task_config:
            import config
            task_config = config.TASK
            
        task_type = task_config.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            return self._majority_voting_yes_no_maybe(agent_responses)
        elif task_type == "multi_choice_mcq":
            return self._borda_count_multi_choice(agent_responses)
        
        # Initialize scores for all valid options
        valid_options = self._get_option_letters(task_config)
        borda_scores = {option: 0 for option in valid_options}
        num_rankings = 0
        max_score = len(valid_options) - 1  # Max points for first place
        
        for agent_role, response_data in agent_responses.items():
            if "final_decision" in response_data:
                # Look for ranking pattern - updated for A-J
                option_pattern = "|".join(valid_options + [opt.lower() for opt in valid_options])
                ranking_pattern = f"[Mm]y\\s+ranking:?\\s*(?:\\*?({option_pattern})\\*?,?\\s*)+"
                
                match = re.search(ranking_pattern, response_data["final_decision"], re.IGNORECASE)
                
                if match:
                    # Extract all options in ranking order
                    options_found = re.findall(f"({option_pattern})", match.group(0), re.IGNORECASE)
                    ranking = [opt.upper() for opt in options_found if opt.upper() in valid_options]
                    
                    if ranking:
                        num_rankings += 1
                        # Assign Borda points
                        for i, option in enumerate(ranking):
                            points = max_score - i
                            borda_scores[option] += max(points, 0)
                        
                        logging.info(f"Borda count: {agent_role} ranking extracted: {ranking}")
                        continue
                
                # Fallback: just extract single answer
                answer = self.extract_answer_option(response_data["final_decision"])
                if answer and answer in valid_options:
                    borda_scores[answer] += max_score
                    num_rankings += 1
                    logging.info(f"Borda count: {agent_role} single answer: {answer}")
        
        # Find winner
        winning_option = max(borda_scores.items(), key=lambda x: x[1])[0] if any(borda_scores.values()) else None
        total_possible_score = num_rankings * max_score
        confidence = borda_scores[winning_option] / total_possible_score if winning_option and total_possible_score > 0 else 0
        
        return {
            "method": "borda_count",
            "winning_option": winning_option,
            "borda_scores": borda_scores,
            "total_possible_score": total_possible_score,
            "confidence": confidence
        }

    def weighted_voting(self, agent_responses, team_weights=None, task_config: Dict[str, Any] = None):
        """Apply weighted voting supporting A-J options."""
        task_config = task_config or self.task_config
        if not task_config:
            import config
            task_config = config.TASK
            
        task_type = task_config.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            return self._weighted_voting_yes_no_maybe(agent_responses, team_weights)
        elif task_type == "multi_choice_mcq":
            return self._weighted_voting_multi_choice(agent_responses, team_weights)
        
        votes = {}
        weighted_votes = {}
        total_weight = 0
        agent_weights_used = {}
        
        # Get valid options
        valid_options = self._get_option_letters(task_config)
        
        for agent_role, response_data in agent_responses.items():
            preference = None
            confidence = 1.0
            
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    preference = self.extract_answer_option(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answer" in response_data["extract"]:
                        preference = response_data["extract"]["answer"]
                        if preference:
                            preference = preference.upper()
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if preference and preference in valid_options:
                weight = response_data.get("weight", 0.2)
                
                # Apply hierarchy multiplier
                hierarchy_factor = 1.0
                if "leader" in agent_role.lower() or "chief" in agent_role.lower():
                    hierarchy_factor = 1.5
                elif any(x in agent_role for x in ["3_", "Final", "FRDT"]):
                    hierarchy_factor = 1.3
                elif any(x in agent_role for x in ["2_", "Expert", "DET"]):
                    hierarchy_factor = 1.1
                
                final_weight = weight * confidence * hierarchy_factor
                agent_weights_used[agent_role] = final_weight
                
                if preference not in votes:
                    votes[preference] = 0
                    weighted_votes[preference] = 0
                
                votes[preference] += 1
                weighted_votes[preference] += final_weight
                total_weight += final_weight
        
        winning_option = None if not weighted_votes else max(weighted_votes.items(), key=lambda x: x[1])[0]
        confidence = weighted_votes.get(winning_option, 0) / total_weight if total_weight > 0 else 0
        
        return {
            "method": "weighted_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "weighted_votes": weighted_votes,
            "agent_weights": agent_weights_used,
            "total_weight": total_weight,
            "confidence": confidence
        }


    # Yes/No/Maybe specific methods
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
                logging.info(f"Recorded vote from {agent_role}: {answer}")
        
        # Find winning option
        total_votes = sum(votes.values())
        winning_option = max(votes, key=votes.get) if total_votes > 0 else None
        
        logging.info(f"Yes/No/Maybe voting results: {votes}, winner: {winning_option}")
        
        return {
            "method": "majority_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "total_votes": total_votes,
            "confidence": votes.get(winning_option, 0) / total_votes if total_votes > 0 else 0
        }
    
    def _weighted_voting_yes_no_maybe(self, agent_responses, team_weights=None):
        """Apply weighted voting to yes/no/maybe tasks."""
        votes = {"yes": 0, "no": 0, "maybe": 0}
        weighted_votes = {"yes": 0.0, "no": 0.0, "maybe": 0.0}
        total_weight = 0
        agent_weights_used = {}
        
        for agent_role, response_data in agent_responses.items():
            answer = None
            confidence = 1.0
            
            # Extract answer and confidence
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    answer = self.extract_yes_no_maybe_answer(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answer" in response_data["extract"]:
                        answer = response_data["extract"]["answer"]
                        if answer and answer.lower() in ["yes", "no", "maybe"]:
                            answer = answer.lower()
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if answer and answer in votes:
                # Get agent weight
                weight = response_data.get("weight", 0.2)
                
                # Apply hierarchy multiplier
                hierarchy_factor = 1.0
                if "leader" in agent_role.lower() or "chief" in agent_role.lower():
                    hierarchy_factor = 1.5
                elif any(x in agent_role for x in ["3_", "Final", "FRDT"]):
                    hierarchy_factor = 1.3
                elif any(x in agent_role for x in ["2_", "Expert", "DET"]):
                    hierarchy_factor = 1.1
                
                # Calculate final weight
                final_weight = weight * confidence * hierarchy_factor
                agent_weights_used[agent_role] = final_weight
                
                # Record vote
                votes[answer] += 1
                weighted_votes[answer] += final_weight
                total_weight += final_weight
                
                logging.info(f"Recorded weighted vote from {agent_role}: {answer} (weight: {final_weight})")
        
        # Find winner
        winning_option = max(weighted_votes, key=weighted_votes.get) if total_weight > 0 else None
        confidence = weighted_votes.get(winning_option, 0) / total_weight if total_weight > 0 else 0
        
        logging.info(f"Yes/No/Maybe weighted voting results: {weighted_votes}, winner: {winning_option}")
        
        return {
            "method": "weighted_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "weighted_votes": weighted_votes,
            "agent_weights": agent_weights_used,
            "total_weight": total_weight,
            "confidence": confidence
        }

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


    def _get_option_letters(self, task_config):
        """Get valid option letters based on task configuration."""
        if task_config and "num_options" in task_config:
            num_options = task_config["num_options"]
            return [chr(65 + i) for i in range(min(num_options, 10))]  # A-J max
        elif task_config and "options" in task_config:
            num_options = len(task_config["options"])
            return [chr(65 + i) for i in range(min(num_options, 10))]  # A-J max
        else:
            return ["A", "B", "C", "D"]  # Default fallback
        

    def extract_answer_option(self, content):
        """Extract answer option supporting A-J options."""
        if not isinstance(content, str):
            return None
        
        # Updated patterns for A-J
        patterns = [
            r"ANSWER:\s*([A-Ja-j])",           # ANSWER: A through J
            r"FINAL ANSWER:\s*([A-Ja-j])",     # FINAL ANSWER: A through J
            r"^ANSWER:\s*([A-Ja-j])",          # Start of line
            r"answer is:?\s*([A-Ja-j])",       # answer is: A
            r"my answer:?\s*([A-Ja-j])",       # my answer: A
            r"the answer:?\s*([A-Ja-j])",      # the answer: A
            r"option\s+([A-Ja-j])",            # option A
            r"choose\s+([A-Ja-j])",            # choose A
            r"select\s+([A-Ja-j])",            # select A
            r"\*\*([A-Ja-j])\.\*\*",          # **A.**
            r"^([A-Ja-j])\.",                  # A. at start of line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        return None
    

    def _weighted_voting_multi_choice(self, agent_responses, team_weights=None):
        """Apply weighted voting to multi-choice MCQ tasks."""
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        weighted_votes = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        total_weight = 0
        agent_weights_used = {}
        
        for agent_role, response_data in agent_responses.items():
            selected_options = []
            confidence = 1.0
            
            # Extract answers and confidence
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    selected_options = self.extract_multi_choice_answers(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answers" in response_data["extract"]:
                        selected_options = response_data["extract"]["answers"]
                    elif "answer" in response_data["extract"]:
                        answer = response_data["extract"]["answer"]
                        if answer:
                            selected_options = [answer.upper()]
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if selected_options:
                # Get agent weight
                weight = response_data.get("weight", 0.2)
                
                # Apply hierarchy multiplier
                hierarchy_factor = 1.0
                if "leader" in agent_role.lower() or "chief" in agent_role.lower():
                    hierarchy_factor = 1.5
                elif any(x in agent_role for x in ["3_", "Final", "FRDT"]):
                    hierarchy_factor = 1.3
                elif any(x in agent_role for x in ["2_", "Expert", "DET"]):
                    hierarchy_factor = 1.1
                
                # Calculate final weight per option
                final_weight = weight * confidence * hierarchy_factor / len(selected_options)
                agent_weights_used[agent_role] = weight * confidence * hierarchy_factor
                
                # Record votes for each selected option
                for option in selected_options:
                    if option in option_votes:
                        option_votes[option] += 1
                        weighted_votes[option] += final_weight
                
                total_weight += weight * confidence * hierarchy_factor
        
        # Determine winning options
        if weighted_votes:
            max_weight = max(weighted_votes.values())
            winning_options = [opt for opt, weight in weighted_votes.items() if weight == max_weight]
        else:
            winning_options = []
        
        # Sort for consistent output
        winning_options = sorted(winning_options)
        
        # Calculate confidence
        confidence = max_weight / total_weight if total_weight > 0 and winning_options else 0
        
        return {
            "method": "weighted_voting_multi_choice",
            "winning_options": winning_options,
            "vote_counts": option_votes,
            "weighted_votes": weighted_votes,
            "agent_weights": agent_weights_used,
            "total_weight": total_weight,
            "confidence": confidence
        }


    def _borda_count_multi_choice(self, agent_responses):
        """Apply modified Borda count to multi-choice MCQ tasks."""
        # For multi-choice, we'll score based on how many times each option is selected
        option_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
        num_responses = 0
        
        for agent_role, response_data in agent_responses.items():
            selected_options = []
            
            if "final_decision" in response_data:
                selected_options = self.extract_multi_choice_answers(response_data["final_decision"])
            
            if selected_options:
                num_responses += 1
                # Give equal points to all selected options
                points_per_option = 3.0 / len(selected_options)
                for option in selected_options:
                    if option in option_scores:
                        option_scores[option] += points_per_option
        
        # Find winning options (those with highest scores)
        if option_scores:
            max_score = max(option_scores.values())
            winning_options = [opt for opt, score in option_scores.items() if score == max_score]
        else:
            winning_options = []
        
        # Sort for consistent output
        winning_options = sorted(winning_options)
        
        # Calculate confidence
        total_possible_score = num_responses * 3.0
        confidence = max_score / total_possible_score if winning_options and total_possible_score > 0 else 0
        
        return {
            "method": "borda_count_multi_choice",
            "winning_options": winning_options,
            "option_scores": option_scores,
            "total_possible_score": total_possible_score,
            "confidence": confidence
        }

        # Yes/No/Maybe specific methods


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
   

        """Apply weighted voting to yes/no/maybe tasks."""
        votes = {"yes": 0, "no": 0, "maybe": 0}
        weighted_votes = {"yes": 0.0, "no": 0.0, "maybe": 0.0}
        total_weight = 0
        agent_weights_used = {}
        
        for agent_role, response_data in agent_responses.items():
            answer = None
            confidence = 1.0
            
            # Extract answer and confidence
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    answer = self.extract_yes_no_maybe_answer(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answer" in response_data["extract"]:
                        answer = response_data["extract"]["answer"]
                        if answer and answer.lower() in ["yes", "no", "maybe"]:
                            answer = answer.lower()
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if answer and answer in votes:
                # Get agent weight
                weight = response_data.get("weight", 0.2)
                
                # Apply hierarchy multiplier
                hierarchy_factor = 1.0
                if "leader" in agent_role.lower() or "chief" in agent_role.lower():
                    hierarchy_factor = 1.5
                elif any(x in agent_role for x in ["3_", "Final", "FRDT"]):
                    hierarchy_factor = 1.3
                elif any(x in agent_role for x in ["2_", "Expert", "DET"]):
                    hierarchy_factor = 1.1
                
                # Calculate final weight
                final_weight = weight * confidence * hierarchy_factor
                agent_weights_used[agent_role] = final_weight
                
                # Record vote
                votes[answer] += 1
                weighted_votes[answer] += final_weight
                total_weight += final_weight
        
        # Find winner
        winning_option = max(weighted_votes, key=weighted_votes.get) if total_weight > 0 else None
        confidence = weighted_votes.get(winning_option, 0) / total_weight if total_weight > 0 else 0
        
        return {
            "method": "weighted_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "weighted_votes": weighted_votes,
            "agent_weights": agent_weights_used,
            "total_weight": total_weight,
            "confidence": confidence
        }