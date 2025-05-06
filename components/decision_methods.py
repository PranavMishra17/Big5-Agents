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
    """
    
    def __init__(self):
        """Initialize the decision methods handler."""
        self.logger = logging.getLogger("decision.methods")
        self.logger.info("Initialized decision methods handler")


    def extract_answer_option(self, content):
        """Extract answer option from various content formats."""
        if not isinstance(content, str):
            return None
        
        # Standard answer formats
        patterns = [
            r"ANSWER:\s*([A-D])",
            r"FINAL ANSWER:\s*([A-D])",
            r"\*\*([A-D])\.",
            r"^([A-D])\.",
            r"option\s+([A-D])",
            r"selected?:?\s*([A-D])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Check for antibody mentions
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


    def majority_voting(self, agent_responses):
        """Apply majority voting to select the most common option."""
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
            # Handle string responses
            elif isinstance(response, str):
                preference = self.extract_answer_option(response)
            
            # Register the vote if preference was found
            if preference:
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


    def borda_count(self, agent_responses):
        """Apply Borda count to MCQ tasks."""
        import re
        
        borda_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
        num_rankings = 0
        
        for agent_role, response_data in agent_responses.items():
            if "final_decision" in response_data:
                # Look for ranking pattern with improved regex
                # Handle bold markdown, optional spaces, and variations
                ranking_pattern = r"[Mm]y\s+ranking:?\s*\*?([A-D])\*?,\s*\*?([A-D])\*?,\s*\*?([A-D])\*?,\s*\*?([A-D])\*?"
                match = re.search(ranking_pattern, response_data["final_decision"], re.IGNORECASE)
                
                if match:
                    # Extract ranking
                    ranking = list(match.groups())
                    num_rankings += 1
                    
                    # Assign Borda points (3 for 1st place, 2 for 2nd, etc.)
                    for i, option in enumerate(ranking):
                        borda_scores[option] += 3 - i  # 3, 2, 1, 0 points
                    
                    logging.info(f"Borda count: {agent_role} ranking extracted: {ranking}")
                else:
                    # Fallback: look for ranking mentioned in a different format
                    alt_pattern = r"ranking.*?([A-D]).*?([A-D]).*?([A-D]).*?([A-D])"
                    alt_match = re.search(alt_pattern, response_data["final_decision"], re.IGNORECASE)
                    
                    if alt_match:
                        ranking = list(alt_match.groups())
                        num_rankings += 1
                        for i, option in enumerate(ranking):
                            borda_scores[option] += 3 - i
                        logging.info(f"Borda count: {agent_role} alt ranking extracted: {ranking}")
                    else:
                        # Second fallback: just extract the answer
                        answer = self.extract_answer_option(response_data["final_decision"])
                        if answer:
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


    def weighted_voting(self, agent_responses, team_weights=None):
        """Apply weighted voting based on agent confidence and hierarchy."""
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
                    if "confidence" in response_data["extract"]:
                        confidence = response_data["extract"]["confidence"]
            
            if preference:
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


    def _majority_voting_ranking(self, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply majority voting to ranking tasks."""
        # For rankings, we'll collect votes for each position
        position_votes = {}
        
        # Initialize vote tracking for each position
        for i in range(len(config.TASK["options"])):
            position_votes[i+1] = {}  # 1-indexed positions
        
        # Count votes for each position
        for agent_role, response_data in agent_responses.items():
            if "ranking" in response_data:
                ranking = response_data["ranking"]
                for i, item in enumerate(ranking):
                    position = i + 1  # 1-indexed position
                    if position in position_votes:
                        if item not in position_votes[position]:
                            position_votes[position][item] = 0
                        position_votes[position][item] += 1
        
        # Determine winning item for each position
        final_ranking = []
        used_items = set()
        
        for position in range(1, len(config.TASK["options"]) + 1):
            # Get votes for this position
            votes = position_votes[position]
            
            # Find the item with the most votes that hasn't been used yet
            max_votes = 0
            winner = None
            
            for item, vote_count in votes.items():
                if item not in used_items and vote_count > max_votes:
                    max_votes = vote_count
                    winner = item
            
            # If no winner (e.g., all voted items already used), find first unused item
            if winner is None:
                for item in config.TASK["options"]:
                    if item not in used_items:
                        winner = item
                        break
            
            if winner:
                final_ranking.append(winner)
                used_items.add(winner)
        
        # Ensure all items are included
        for item in config.TASK["options"]:
            if item not in final_ranking:
                final_ranking.append(item)
        
        # Calculate the number of agents that had the same top item
        top_item = final_ranking[0] if final_ranking else None
        top_votes = sum(1 for response in agent_responses.values() 
                      if "ranking" in response and response["ranking"] 
                      and response["ranking"][0] == top_item)
        
        return {
            "method": "majority_voting",
            "final_ranking": final_ranking,
            "top_item": top_item,
            "top_votes": top_votes,
            "total_votes": len(agent_responses),
            "confidence": top_votes / max(1, len(agent_responses))
        }
    

    def _majority_voting_mcq(self, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply majority voting to MCQ tasks."""
        vote_counts = {}
        
        # Count votes for each option
        for agent_role, response_data in agent_responses.items():
            if "answer" in response_data:
                answer = response_data["answer"]
                if answer not in vote_counts:
                    vote_counts[answer] = 0
                vote_counts[answer] += 1
        
        # Find the option with the most votes
        max_votes = 0
        winner = None
        
        for option, count in vote_counts.items():
            if count > max_votes:
                max_votes = count
                winner = option
        
        # If no clear winner, use the first agent's response
        if winner is None and agent_responses:
            first_agent = list(agent_responses.keys())[0]
            if "answer" in agent_responses[first_agent]:
                winner = agent_responses[first_agent]["answer"]
        
        return {
            "method": "majority_voting",
            "winning_option": winner,
            "vote_counts": vote_counts,
            "total_votes": len(agent_responses),
            "confidence": max_votes / max(1, len(agent_responses)) if max_votes > 0 else 0
        }
    

    def _majority_voting_general(self, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply majority voting to general tasks."""
        # For general tasks, we use a simple consensus approach
        # This is more qualitative since there's no clear "vote"
        
        # Collect all responses
        responses = []
        confidences = []
        
        for agent_role, response_data in agent_responses.items():
            if "response" in response_data:
                responses.append(response_data["response"])
                if "confidence" in response_data:
                    confidences.append(response_data["confidence"])
        
        # For general tasks, we'll use the highest confidence response
        if responses and confidences:
            max_confidence_idx = confidences.index(max(confidences))
            consensus_response = responses[max_confidence_idx]
            confidence = confidences[max_confidence_idx]
        elif responses:
            # If no confidence values, use the first response
            consensus_response = responses[0]
            confidence = 0.5  # Default medium confidence
        else:
            consensus_response = "No consensus reached."
            confidence = 0
        
        return {
            "method": "majority_voting",
            "consensus_response": consensus_response,
            "num_responses": len(responses),
            "confidence": confidence
        }


    def _weighted_voting_ranking(self, agent_responses: Dict[str, Dict[str, Any]], 
                               agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """Apply weighted voting to ranking tasks."""
        # For rankings, we'll assign weighted votes to items at each position
        position_votes = {}
        
        # Initialize vote tracking for each position
        for i in range(len(config.TASK["options"])):
            position_votes[i+1] = {}  # 1-indexed positions
        
        # Track the weighted votes
        for agent_role, response_data in agent_responses.items():
            if "ranking" in response_data:
                ranking = response_data["ranking"]
                
                # Get agent weight and confidence
                weight = agent_weights.get(agent_role, 1.0)
                confidence = response_data.get("confidence", 0.7)  # Default confidence if not provided
                
                # Combined weight based on agent expertise and confidence
                combined_weight = weight * confidence
                
                for i, item in enumerate(ranking):
                    position = i + 1  # 1-indexed position
                    if position in position_votes:
                        if item not in position_votes[position]:
                            position_votes[position][item] = 0
                        position_votes[position][item] += combined_weight
        
        # Determine winning item for each position
        final_ranking = []
        used_items = set()
        position_winners = {}
        
        for position in range(1, len(config.TASK["options"]) + 1):
            # Get votes for this position
            votes = position_votes[position]
            
            # Find the item with the most weighted votes that hasn't been used yet
            max_votes = 0
            winner = None
            
            for item, vote_weight in votes.items():
                if item not in used_items and vote_weight > max_votes:
                    max_votes = vote_weight
                    winner = item
            
            # If no winner (e.g., all voted items already used), find first unused item
            if winner is None:
                for item in config.TASK["options"]:
                    if item not in used_items:
                        winner = item
                        break
            
            if winner:
                final_ranking.append(winner)
                used_items.add(winner)
                position_winners[position] = {"item": winner, "weight": max_votes}
        
        # Ensure all items are included
        for item in config.TASK["options"]:
            if item not in final_ranking:
                final_ranking.append(item)
        
        # Calculate total weight and confidence
        total_weight = sum(agent_weights.get(role, 1.0) * response.get("confidence", 0.7) 
                          for role, response in agent_responses.items())
        
        # Weight of the top item
        top_item = final_ranking[0] if final_ranking else None
        top_weight = position_winners.get(1, {}).get("weight", 0)
        
        return {
            "method": "weighted_voting",
            "final_ranking": final_ranking,
            "top_item": top_item,
            "top_weight": top_weight,
            "total_weight": total_weight,
            "confidence": top_weight / max(0.001, total_weight),
            "position_weights": position_winners
        }
    


    def _weighted_voting_mcq(self, agent_responses: Dict[str, Dict[str, Any]], 
                        agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """Apply weighted voting to MCQ tasks."""
        weighted_votes = {}
        
        # Count weighted votes for each option
        for agent_role, response_data in agent_responses.items():
            if "answer" in response_data:
                answer = response_data["answer"]
                if answer not in weighted_votes:
                    weighted_votes[answer] = 0
                
                # Get agent weight and confidence
                base_weight = agent_weights.get(agent_role, 1.0)
                confidence = response_data.get("confidence", 0.7)  # Default confidence if not provided
                
                # Apply hierarchical weight adjustment
                hierarchy_multiplier = 1.0
                
                # Check if this is a leader/chief role
                if "lead" in agent_role.lower() or "chief" in agent_role.lower():
                    hierarchy_multiplier = 1.5
                # Check if this is from a final decision team
                elif "final" in agent_role.lower() or "decision" in agent_role.lower() or "frdt" in agent_role.lower():
                    hierarchy_multiplier = 1.3
                # Check if this is a subordinate role
                elif "subordinate" in response_data.get("hierarchy", "").lower():
                    hierarchy_multiplier = 0.8
                
                # Combined weight based on agent expertise, confidence, and hierarchy
                weighted_votes[answer] += base_weight * confidence * hierarchy_multiplier
        
        # Find the option with the most weighted votes
        max_votes = 0
        winner = None
        
        for option, weight in weighted_votes.items():
            if weight > max_votes:
                max_votes = weight
                winner = option
        
        # Calculate total weight for confidence calculation
        total_weight = sum(agent_weights.get(role, 1.0) * response.get("confidence", 0.7) 
                        for role, response in agent_responses.items())
        
        return {
            "method": "weighted_voting",
            "winning_option": winner,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight,
            "confidence": max_votes / max(0.001, total_weight) if max_votes > 0 else 0
        }


    def _weighted_voting_general(self, agent_responses: Dict[str, Dict[str, Any]], 
                               agent_weights: Dict[str, float]) -> Dict[str, Any]:
        """Apply weighted voting to general tasks."""
        # For general tasks, we select the response from the agent with highest weighted confidence
        
        # Track responses and their weighted confidence
        weighted_confidences = []
        responses = []
        
        for agent_role, response_data in agent_responses.items():
            if "response" in response_data:
                # Get agent weight and confidence
                weight = agent_weights.get(agent_role, 1.0)
                confidence = response_data.get("confidence", 0.7)  # Default confidence if not provided
                
                # Combined weight based on agent expertise and confidence
                weighted_confidences.append(weight * confidence)
                responses.append(response_data["response"])
        
        # Select the response with the highest weighted confidence
        if responses and weighted_confidences:
            max_confidence_idx = weighted_confidences.index(max(weighted_confidences))
            selected_response = responses[max_confidence_idx]
            confidence = weighted_confidences[max_confidence_idx]
        elif responses:
            # If no confidence values, use the first response
            selected_response = responses[0]
            confidence = 0.5  # Default medium confidence
        else:
            selected_response = "No weighted consensus reached."
            confidence = 0
        
        # Calculate total weight for normalization
        total_weight = sum(agent_weights.get(role, 1.0) for role in agent_responses.keys())
        
        return {
            "method": "weighted_voting",
            "selected_response": selected_response,
            "num_responses": len(responses),
            "weighted_confidence": confidence,
            "normalized_confidence": confidence / max(0.001, total_weight)
        }
    

    def _borda_count_ranking(self, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Borda count to ranking tasks."""
        n_items = len(config.TASK["options"])
        borda_scores = {item: 0 for item in config.TASK["options"]}
        
        # Calculate Borda scores for each item
        for agent_role, response_data in agent_responses.items():
            if "ranking" in response_data:
                ranking = response_data["ranking"]
                for i, item in enumerate(ranking):
                    # In Borda count, higher positions get more points
                    # For a ranking task, the first position gets n-1 points, 
                    # second gets n-2, etc.
                    borda_scores[item] += (n_items - i - 1)
        
        # Sort items by Borda score in descending order
        final_ranking = sorted(borda_scores.keys(), 
                              key=lambda item: borda_scores[item], 
                              reverse=True)
        
        # Calculate normalized confidence based on score distribution
        total_possible_score = len(agent_responses) * (n_items * (n_items - 1) / 2)
        total_actual_score = sum(borda_scores.values())
        
        # The entropy of the score distribution gives an idea of consensus
        # Lower entropy means more agreement
        score_entropy = 0
        if total_actual_score > 0:
            for item in config.TASK["options"]:
                p = borda_scores[item] / total_actual_score
                if p > 0:
                    score_entropy -= p * math.log2(p)
            
            # Normalize entropy to 0-1 range (1 being perfect consensus)
            max_entropy = math.log2(n_items)
            normalized_entropy = 1 - (score_entropy / max_entropy)
        else:
            normalized_entropy = 0
        
        return {
            "method": "borda_count",
            "final_ranking": final_ranking,
            "borda_scores": borda_scores,
            "top_item": final_ranking[0] if final_ranking else None,
            "top_score": borda_scores[final_ranking[0]] if final_ranking else 0,
            "total_possible_score": total_possible_score,
            "confidence": normalized_entropy
        }
    

    def _borda_count_mcq(self, agent_responses):
        """Apply Borda count to MCQ tasks."""
        import re
        
        borda_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
        num_rankings = 0
        
        for agent_role, response_data in agent_responses.items():
            if "final_decision" in response_data:
                # Look for ranking pattern with improved regex
                # Handle bold markdown, optional spaces, and variations
                ranking_pattern = r"[Mm]y\s+ranking:?\s*\*?([A-D])\*?,\s*\*?([A-D])\*?,\s*\*?([A-D])\*?,\s*\*?([A-D])\*?"
                match = re.search(ranking_pattern, response_data["final_decision"], re.IGNORECASE)
                
                if match:
                    # Extract ranking
                    ranking = list(match.groups())
                    num_rankings += 1
                    
                    # Assign Borda points (3 for 1st place, 2 for 2nd, etc.)
                    for i, option in enumerate(ranking):
                        borda_scores[option] += 3 - i  # 3, 2, 1, 0 points
                    
                    logging.info(f"Borda count: {agent_role} ranking extracted: {ranking}")
                else:
                    # Fallback: look for ranking mentioned in a different format
                    alt_pattern = r"ranking.*?([A-D]).*?([A-D]).*?([A-D]).*?([A-D])"
                    alt_match = re.search(alt_pattern, response_data["final_decision"], re.IGNORECASE)
                    
                    if alt_match:
                        ranking = list(alt_match.groups())
                        num_rankings += 1
                        for i, option in enumerate(ranking):
                            borda_scores[option] += 3 - i
                        logging.info(f"Borda count: {agent_role} alt ranking extracted: {ranking}")
                    else:
                        # Second fallback: just extract the answer
                        answer = self.extract_answer_option(response_data["final_decision"])
                        if answer:
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


    def _borda_count_general(self, agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Borda count to general tasks."""
        # For general tasks, Borda count isn't directly applicable
        # We'll use a modified approach based on confidence levels
        
        # Collect responses and confidences
        responses = []
        confidences = []
        
        for agent_role, response_data in agent_responses.items():
            if "response" in response_data:
                responses.append(response_data["response"])
                confidences.append(response_data.get("confidence", 0.7))  # Default confidence if not provided
        
        # Rank responses by confidence
        ranked_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        ranked_responses = [responses[i] for i in ranked_indices]
        ranked_confidences = [confidences[i] for i in ranked_indices]
        
        # The response with highest confidence is the winner
        top_response = ranked_responses[0] if ranked_responses else "No response available."
        
        # Calculate confidence based on distribution of confidence scores
        confidence_sum = sum(confidences)
        normalized_confidence = ranked_confidences[0] / max(0.001, confidence_sum) if ranked_confidences else 0
        
        return {
            "method": "borda_count",
            "selected_response": top_response,
            "confidence_ranking": list(zip(ranked_responses, ranked_confidences)) if ranked_responses else [],
            "normalized_confidence": normalized_confidence
        }