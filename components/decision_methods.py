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
    
    Supports multiple decision methods:
    1. Majority Voting - Each agent gets one vote with equal weight
    2. Weighted Voting - Votes are weighted by agent expertise and confidence
    3. Borda Count - Points assigned based on preference ranking
    4. Yes/No/Maybe Voting - For research questions requiring binary or uncertain answers
    """
    
    def __init__(self):
        """Initialize the decision methods handler."""
        self.logger = logging.getLogger("decision.methods")
        self.logger.info("Initialized decision methods handler")

    def extract_answer_option(self, content):
        """Extract answer option from various content formats."""
        if not isinstance(content, str):
            return None
        
        # Standard answer formats for MCQ
        patterns = [
            r"ANSWER:\s*([A-Da-d])",
            r"FINAL ANSWER:\s*([A-Da-d])",
            r"\*\*([A-Da-d])\.",
            r"^([A-Da-d])\.",
            r"option\s+([A-Da-d])",
            r"selected?:?\s*([A-Da-d])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Always return uppercase 
                return match.group(1).upper()
        
        # Check for antibody mentions (specific to some medical questions)
        antibody_map = {
            "A": ["anti-nmda", "nmda receptor"],
            "B": ["anti-lgi1", "lgi1"],
            "C": ["anti-gaba", "gaba-b"],
            "D": ["anti-ampa", "ampa receptor"]
        }
        
        for option, keywords in antibody_map.items():
            if any(keyword in content.lower() for keyword in keywords):
                return option
                
        return None
    
    def extract_yes_no_maybe_answer(self, content):
        """Extract yes/no/maybe answer from content."""
        if not isinstance(content, str):
            return None
        
        content_lower = content.lower()
        
        # Explicit answer patterns
        patterns = [
            r"ANSWER:\s*(yes|no|maybe)",
            r"FINAL ANSWER:\s*(yes|no|maybe)",
            r"answer is:\s*(yes|no|maybe)",
            r"my answer:\s*(yes|no|maybe)",
            r"the answer is:\s*(yes|no|maybe)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # Check for clear yes/no/maybe statements
        if "yes," in content_lower or "yes." in content_lower or "yes\n" in content_lower:
            return "yes"
        elif "no," in content_lower or "no." in content_lower or "no\n" in content_lower:
            return "no"
        elif "maybe" in content_lower or "uncertain" in content_lower or "possibly" in content_lower:
            return "maybe"
        
        return None

    def extract_multi_choice_answers(self, content):
        """Extract multiple choice answers from content (e.g., A,C or A,B,D)."""
        if not isinstance(content, str):
            return []
        
        import re
        
        # Patterns for multi-choice answers
        patterns = [
            r"ANSWERS?:\s*([A-D](?:\s*,\s*[A-D])*)",
            r"FINAL ANSWERS?:\s*([A-D](?:\s*,\s*[A-D])*)",
            r"selected options?:?\s*([A-D](?:\s*,\s*[A-D])*)",
            r"correct options? (?:are|is):?\s*([A-D](?:\s*,\s*[A-D])*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract and clean the answer string
                answer_str = match.group(1).upper()
                # Remove spaces and split by comma
                answers = [a.strip() for a in answer_str.split(',') if a.strip()]
                # Remove duplicates and sort
                return sorted(list(set(answers)))
        
        # Fallback: look for pattern like "Options A and C" or "A, B, and D"
        option_pattern = r"options?\s+([A-D](?:\s*,\s*[A-D])*(?:\s*,?\s*and\s*[A-D])?)"
        match = re.search(option_pattern, content, re.IGNORECASE)
        if match:
            answer_str = match.group(1).upper()
            answer_str = answer_str.replace('AND', ',').replace(' ', '')
            answers = [a.strip() for a in answer_str.split(',') if a.strip()]
            return sorted(list(set(answers)))
        
        return []

    def multi_choice_voting(self, agent_responses):
        """Apply voting to multi-choice MCQ tasks."""
        # Count votes for each option
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        
        for agent_role, response in agent_responses.items():
            selected_options = []
            
            # Handle dictionary responses
            if isinstance(response, dict):
                if "final_decision" in response:
                    selected_options = self.extract_multi_choice_answers(response["final_decision"])
                elif "extract" in response and isinstance(response["extract"], dict):
                    if "answers" in response["extract"]:
                        selected_options = response["extract"]["answers"]
                    elif "answer" in response["extract"]:
                        # Single answer, convert to list
                        answer = response["extract"]["answer"]
                        if answer:
                            selected_options = [answer.upper()]
            # Handle string responses
            elif isinstance(response, str):
                selected_options = self.extract_multi_choice_answers(response)
            
            # Register votes for each selected option
            for option in selected_options:
                if option in option_votes:
                    option_votes[option] += 1
        
        # Determine winning options (those with more than 50% votes)
        total_voters = len(agent_responses)
        threshold = total_voters / 2
        winning_options = [opt for opt, votes in option_votes.items() if votes > threshold]
        
        # If no option has majority, take the top voted options
        if not winning_options and option_votes:
            max_votes = max(option_votes.values())
            winning_options = [opt for opt, votes in option_votes.items() if votes == max_votes]
        
        # Sort for consistent output
        winning_options = sorted(winning_options)
        
        # Calculate confidence based on vote distribution
        confidence = 0.0
        if winning_options and total_voters > 0:
            avg_votes = sum(option_votes[opt] for opt in winning_options) / len(winning_options)
            confidence = avg_votes / total_voters
        
        return {
            "method": "multi_choice_voting",
            "winning_options": winning_options,
            "vote_counts": option_votes,
            "total_voters": total_voters,
            "confidence": confidence
        }

    def majority_voting(self, agent_responses):
        """Apply majority voting to select the most common option."""
        task_type = config.TASK.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            return self._majority_voting_yes_no_maybe(agent_responses)
        elif task_type == "multi_choice_mcq":
            return self.multi_choice_voting(agent_responses)
        
        votes = {}
        max_votes = 0
        
        for agent_role, response in agent_responses.items():
            preference = None
            
            # Handle dictionary responses
            if isinstance(response, dict):
                if "final_decision" in response:
                    preference = self.extract_answer_option(response["final_decision"])
                elif "extract" in response and isinstance(response["extract"], dict):
                    if "answer" in response["extract"]:
                        preference = response["extract"]["answer"]
                        if preference:  # Ensure it's not None
                            preference = preference.upper()  # Convert to uppercase
            # Handle string responses
            elif isinstance(response, str):
                preference = self.extract_answer_option(response)
            
            # Register the vote if preference was found
            if preference:
                # Ensure uppercase
                preference = preference.upper()
                
                if preference not in votes:
                    votes[preference] = 0
                votes[preference] += 1
                max_votes = max(max_votes, votes[preference])
        
        # Find winning option
        winning_option = None if not votes else max(votes, key=votes.get)
        total_votes = sum(votes.values())
        
        return {
            "method": "majority_voting",
            "winning_option": winning_option,
            "vote_counts": votes,
            "total_votes": total_votes,
            "confidence": max_votes / total_votes if total_votes > 0 else 0
        }

    def weighted_voting(self, agent_responses, team_weights=None):
        """Apply weighted voting based on agent confidence and hierarchy."""
        task_type = config.TASK.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            return self._weighted_voting_yes_no_maybe(agent_responses, team_weights)
        elif task_type == "multi_choice_mcq":
            return self._weighted_voting_multi_choice(agent_responses, team_weights)
        
        votes = {}
        weighted_votes = {}
        total_weight = 0
        agent_weights_used = {}
        
        for agent_role, response_data in agent_responses.items():
            preference = None
            confidence = 1.0
            
            # Extract answer and confidence
            if isinstance(response_data, dict):
                if "final_decision" in response_data:
                    preference = self.extract_answer_option(response_data["final_decision"])
                elif "extract" in response_data and isinstance(response_data["extract"], dict):
                    if "answer" in response_data["extract"]:
                        preference = response_data["extract"]["answer"]
                        if preference:  # Ensure it's not None
                            preference = preference.upper()  # Convert to uppercase
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if preference:
                # Ensure uppercase
                preference = preference.upper()
                
                # Get agent weight from response data or use default
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
                if preference not in votes:
                    votes[preference] = 0
                    weighted_votes[preference] = 0
                
                votes[preference] += 1
                weighted_votes[preference] += final_weight
                total_weight += final_weight
        
        # Log weights used
        logging.info(f"Agent weights used in voting: {agent_weights_used}")
        
        # Find winner
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

    def borda_count(self, agent_responses):
        """Apply Borda count to MCQ tasks."""
        task_type = config.TASK.get("type", "mcq")
        
        if task_type == "yes_no_maybe":
            # Borda count doesn't apply well to yes/no/maybe, so use majority voting
            return self._majority_voting_yes_no_maybe(agent_responses)
        elif task_type == "multi_choice_mcq":
            # For multi-choice, use a modified approach
            return self._borda_count_multi_choice(agent_responses)
        
        import re
        
        # Initialize scores with uppercase letters
        borda_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
        num_rankings = 0
        
        for agent_role, response_data in agent_responses.items():
            if "final_decision" in response_data:
                # Look for ranking pattern with improved regex
                # Handle bold markdown, optional spaces, and variations
                ranking_pattern = r"[Mm]y\s+ranking:?\s*\*?([A-Da-d])\*?,\s*\*?([A-Da-d])\*?,\s*\*?([A-Da-d])\*?,\s*\*?([A-Da-d])\*?"
                match = re.search(ranking_pattern, response_data["final_decision"], re.IGNORECASE)
                
                if match:
                    # Extract ranking, and convert all to uppercase
                    ranking = [option.upper() for option in match.groups()]
                    num_rankings += 1
                    
                    # Assign Borda points (3 for 1st place, 2 for 2nd, etc.)
                    for i, option in enumerate(ranking):
                        borda_scores[option] += 3 - i  # 3, 2, 1, 0 points
                    
                    logging.info(f"Borda count: {agent_role} ranking extracted: {ranking}")
                else:
                    # Fallback: look for ranking mentioned in a different format
                    alt_pattern = r"ranking.*?([A-Da-d]).*?([A-Da-d]).*?([A-Da-d]).*?([A-Da-d])"
                    alt_match = re.search(alt_pattern, response_data["final_decision"], re.IGNORECASE)
                    
                    if alt_match:
                        ranking = [option.upper() for option in alt_match.groups()]
                        num_rankings += 1
                        for i, option in enumerate(ranking):
                            borda_scores[option] += 3 - i
                        logging.info(f"Borda count: {agent_role} alt ranking extracted: {ranking}")
                    else:
                        # Second fallback: just extract the answer
                        answer = self.extract_answer_option(response_data["final_decision"])
                        if answer:
                            # Make sure answer is uppercase
                            answer = answer.upper()
                            borda_scores[answer] += 3
                            num_rankings += 1
                            logging.info(f"Borda count: {agent_role} only answer found: {answer}")
        
        # Find winner and calculate confidence
        winning_option = max(borda_scores.items(), key=lambda x: x[1])[0] if borda_scores else None
        total_possible_score = num_rankings * 3.0
        confidence = borda_scores[winning_option] / total_possible_score if winning_option and total_possible_score > 0 else 0
        
        return {
            "method": "borda_count",
            "winning_option": winning_option,
            "borda_scores": borda_scores,
            "total_possible_score": total_possible_score,
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