"""
Simulator for running agent system interactions.
"""

import os
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time

from components.modular_agent import ModularAgent, create_agent_team
from components.closed_loop import ClosedLoopCommunication
from components.mutual_monitoring import MutualMonitoring
from components.shared_mental_model import SharedMentalModel
from components.decision_methods import DecisionMethods
from utils.logger import SimulationLogger
import config
from components.team_orientation import TeamOrientation
from components.mutual_trust import MutualTrust

from utils.prompts import DISCUSSION_PROMPTS



class AgentSystemSimulator:
    """
    Simulator for running agent system with modular teamwork components.
    """
    
    # Update the AgentSystemSimulator __init__ method to support recruitment
    def __init__(self, 
             simulation_id: str = None,
             use_team_leadership: bool = None,
             use_closed_loop_comm: bool = None,
             use_mutual_monitoring: bool = None,
             use_shared_mental_model: bool = None,
             use_team_orientation: bool = None,
             use_mutual_trust: bool = None,
             mutual_trust_factor: float = None,
             random_leader: bool = False,
             use_recruitment: bool = None,
             recruitment_method: str = None,
             recruitment_pool: str = None):
        """
        Initialize the simulator.
        
        Args:
            simulation_id: Optional ID for the simulation, defaults to timestamp
            use_team_leadership: Whether to use team leadership behaviors
            use_closed_loop_comm: Whether to use closed-loop communication
            use_mutual_monitoring: Whether to use mutual performance monitoring
            use_shared_mental_model: Whether to use shared mental models
            use_team_orientation: Whether to use team orientation
            use_mutual_trust: Whether to use mutual trust
            mutual_trust_factor: Trust factor for mutual trust (0.0-1.0)
            random_leader: Whether to randomly assign leadership
            use_recruitment: Whether to use dynamic agent recruitment
            recruitment_method: Method for recruitment (adaptive, fixed, basic, intermediate, advanced)
            recruitment_pool: Pool of agent roles to recruit from
        """
        # Set simulation ID and configuration
        self.simulation_id = simulation_id or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use config values if not specified
        self.use_team_leadership = use_team_leadership if use_team_leadership is not None else config.USE_TEAM_LEADERSHIP
        self.use_closed_loop_comm = use_closed_loop_comm if use_closed_loop_comm is not None else config.USE_CLOSED_LOOP_COMM
        self.use_mutual_monitoring = use_mutual_monitoring if use_mutual_monitoring is not None else config.USE_MUTUAL_MONITORING
        self.use_shared_mental_model = use_shared_mental_model if use_shared_mental_model is not None else config.USE_SHARED_MENTAL_MODEL
        self.use_team_orientation = use_team_orientation if use_team_orientation is not None else config.USE_TEAM_ORIENTATION
        self.use_mutual_trust = use_mutual_trust if use_mutual_trust is not None else config.USE_MUTUAL_TRUST
        self.mutual_trust_factor = mutual_trust_factor if mutual_trust_factor is not None else config.MUTUAL_TRUST_FACTOR
        self.use_recruitment = use_recruitment if use_recruitment is not None else config.USE_AGENT_RECRUITMENT
        self.recruitment_method = recruitment_method or config.RECRUITMENT_METHOD
        self.recruitment_pool = recruitment_pool or "general"
        self.random_leader = random_leader
        self.metadata = {}
        if use_recruitment and recruitment_method:
            self.metadata["complexity"] = recruitment_method
        
        # Setup configuration
        self.config = {
            "use_team_leadership": self.use_team_leadership,
            "use_closed_loop_comm": self.use_closed_loop_comm,
            "use_mutual_monitoring": self.use_mutual_monitoring,
            "use_shared_mental_model": self.use_shared_mental_model,
            "use_team_orientation": self.use_team_orientation,
            "use_mutual_trust": self.use_mutual_trust,
            "mutual_trust_factor": self.mutual_trust_factor,
            "use_recruitment": self.use_recruitment,
            "recruitment_method": self.recruitment_method,
            "recruitment_pool": self.recruitment_pool,
            "random_leader": self.random_leader,
            "task": config.TASK["name"]
        }
        
        # Setup logging
        self.logger = SimulationLogger(
            simulation_id=self.simulation_id,
            log_dir=config.LOG_DIR,
            config=self.config
        )
        
        # Create agent team
        team_data = create_agent_team(
            use_team_leadership=self.use_team_leadership,
            use_closed_loop_comm=self.use_closed_loop_comm,
            use_mutual_monitoring=self.use_mutual_monitoring,
            use_shared_mental_model=self.use_shared_mental_model,
            use_team_orientation=self.use_team_orientation,
            use_mutual_trust=self.use_mutual_trust,
            random_leader=self.random_leader,
            use_recruitment=self.use_recruitment,
            question=config.TASK["description"] if self.use_recruitment else None,
            recruitment_method=self.recruitment_method,
            recruitment_pool=self.recruitment_pool
        )
        
        # Handle returned team data structure
        if isinstance(team_data, dict) and "agents" in team_data and "leader" in team_data:
            self.agents = team_data["agents"]
            self.leader = team_data["leader"]
        else:
            # Unpack tuple return value (agents, leader)
            self.agents, self.leader = team_data
        
        # Initialize teamwork components
        self.comm_handler = ClosedLoopCommunication() if self.use_closed_loop_comm else None
        self.mutual_monitor = MutualMonitoring() if self.use_mutual_monitoring else None
        self.mental_model = SharedMentalModel() if self.use_shared_mental_model else None
        self.team_orientation = TeamOrientation() if self.use_team_orientation else None
        self.mutual_trust = MutualTrust(self.mutual_trust_factor) if self.use_mutual_trust else None
        
        # Initialize mutual trust network if enabled
        if self.mutual_trust:
            self.mutual_trust.initialize_trust_network(list(self.agents.keys()))
        
        # Initialize decision methods
        self.decision_methods = DecisionMethods()
        
        # Initialize shared knowledge
        if self.mental_model:
            # Initialize task model
            self.mental_model.initialize_task_model(config.TASK)
            
            # Initialize team model
            self.mental_model.initialize_team_model(list(self.agents.keys()))
        
        # Store results
        self.results = {
            "simulation_id": self.simulation_id,
            "config": self.config,
            "exchanges": [],
            "decision_results": {
                "majority_voting": None,
                "weighted_voting": None,
                "borda_count": None
            },
            "teamwork_metrics": {}
        }
        
        self.logger.logger.info(f"Initialized simulation {self.simulation_id}")
        
        # Log configuration
        component_config = []
        if self.use_team_leadership:
            leader_role = self.leader.role if self.leader else "None"
            component_config.append(f"Team Leadership: {leader_role}")
        if self.use_closed_loop_comm:
            component_config.append("Closed-loop Communication")
        if self.use_mutual_monitoring:
            component_config.append("Mutual Performance Monitoring")
        if self.use_shared_mental_model:
            component_config.append("Shared Mental Model")
        if self.use_team_orientation:
            component_config.append("Team Orientation")
        if self.use_mutual_trust:
            component_config.append(f"Mutual Trust (factor: {self.mutual_trust_factor:.1f})")
        if self.use_recruitment:
            component_config.append(f"Agent Recruitment ({self.recruitment_method})")
            
        config_str = ", ".join(component_config) if component_config else "No teamwork components"
        self.logger.logger.info(f"Components enabled: {config_str}")


    def run_simulation(self):
        """
        Run the full simulation process with all three decision methods.
        
        Returns:
            Dictionary with simulation results
        """
        # Begin with initial task analysis by each agent
        self.logger.logger.info("Starting task analysis phase")
        agent_analyses = self._run_task_analysis()
        
        # If leadership is enabled, have leader define the team's approach
        if self.use_team_leadership and self.leader:
            self.logger.logger.info("Leadership phase: Defining task approach")
            self._run_leadership_definition()

        # Add complexity metrics to the results if recruitment was used
        if self.use_recruitment:
            from components.agent_recruitment import complexity_counts
            self.results["recruitment_metrics"] = {
                "complexity_distribution": complexity_counts,
                "total_questions": sum(complexity_counts.values()),
                "method": self.recruitment_method,
                "pool": self.recruitment_pool
            }
        
        # Run collaborative discussion
        self.logger.logger.info("Starting collaborative discussion phase")
        agent_decisions = self._run_collaborative_discussion(agent_analyses)
        
        # Apply decision methods
        self.logger.logger.info("Applying decision methods")
        decision_results = self._apply_decision_methods(agent_decisions)
        
        # Store results
        self.results["decision_results"] = decision_results
        
        # Add teamwork metrics if components were used
        if self.use_closed_loop_comm and self.comm_handler:
            self.results["teamwork_metrics"]["closed_loop_communication"] = self.comm_handler.get_communication_metrics()
            self.logger.logger.info("Closed-loop communication metrics collected")
            
        if self.use_mutual_monitoring and self.mutual_monitor:
            self.results["teamwork_metrics"]["mutual_monitoring"] = self.mutual_monitor.analyze_team_performance()
            self.logger.logger.info("Mutual monitoring metrics collected")
            
        if self.use_shared_mental_model and self.mental_model:
            self.results["teamwork_metrics"]["shared_mental_model"] = self.mental_model.analyze_mental_model_effectiveness()
            self.logger.logger.info("Shared mental model metrics collected")
        
        # Save results
        self.save_results()
        
        return self.results
    
    
    def _run_task_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Run initial task analysis by each agent.
        
        Returns:
            Dictionary mapping agent roles to their analyses
        """
        agent_analyses = {}
        
        for role, agent in self.agents.items():
            self.logger.logger.info(f"Getting initial analysis from {role}")
            
            # Enhance prompt with shared mental model if enabled
            if self.use_shared_mental_model and self.mental_model:
                analysis = agent.analyze_task()
                
                # Extract understanding and update shared mental model
                understanding = self.mental_model.extract_understanding_from_message(analysis)
                self.mental_model.update_shared_understanding(role, understanding)
                
                # Log to shared mental model channel
                self.logger.log_mental_model_update(
                    role,
                    understanding,
                    self.mental_model.convergence_metrics[-1]["overall_convergence"] if self.mental_model.convergence_metrics else 0.0
                )
            else:
                analysis = agent.analyze_task()
            
            # Log to main discussion channel
            self.logger.log_main_discussion(
                "task_analysis",
                role,
                analysis
            )
            
            # Store analysis
            agent_analyses[role] = {
                "analysis": analysis,
                "extract": agent.extract_response(analysis)
            }
        
        return agent_analyses


    def _run_leadership_definition(self) -> str:
        """
        Have the leader define the team's approach.
        
        Returns:
            Leader's definition of the task approach
        """
        if not self.leader:
            return "No leader designated for this simulation."
        
        # Have leader define the task approach
        leadership_prompt = f"""
        As the leader, you should define the team's overall approach to solving this task:
        
        {config.TASK['description']}
        
        Please:
        1. Break down this task into clear steps
        2. Define how the team should work together
        3. Specify what each team member should contribute based on their expertise
        4. Outline how we will reach a final decision
        
        Provide clear, specific guidance that will help the team work effectively together.
        """
        
        # Use closed-loop communication if enabled
        if self.use_closed_loop_comm and self.comm_handler:
            # Select first non-leader agent to respond
            responder_role = next((role for role in self.agents if role != self.leader.role), None)
            
            if responder_role:
                responder = self.agents[responder_role]
                
                # Facilitate closed-loop exchange
                exchange = self.comm_handler.facilitate_exchange(
                    self.leader,
                    responder,
                    leadership_prompt
                )
                
                # Log to closed-loop channel
                self.logger.log_closed_loop(
                    "leader_definition",
                    self.leader.role,
                    responder_role,
                    exchange[0],
                    exchange[1],
                    exchange[2]
                )
                
                # Store in results
                self.results["exchanges"].append({
                    "type": "leadership_definition",
                    "communication": "closed_loop",
                    "sender": self.leader.role,
                    "receiver": responder_role,
                    "initial_message": exchange[0],
                    "acknowledgment": exchange[1],
                    "verification": exchange[2]
                })
                
                # Log to leadership channel
                self.logger.log_leadership_action(
                    "task_definition",
                    exchange[0]
                )
                
                return exchange[0]
        else:
            # Standard communication
            leader_definition = self.leader.leadership_action("define_task")
            
            # Log to main discussion channel
            self.logger.log_main_discussion(
                "leadership_definition",
                self.leader.role,
                leader_definition
            )
            
            # Log to leadership channel
            self.logger.log_leadership_action(
                "task_definition",
                leader_definition
            )
            
            # Store in results
            self.results["exchanges"].append({
                "type": "leadership_definition",
                "communication": "standard",
                "sender": self.leader.role,
                "message": leader_definition
            })
            
            return leader_definition
    

    def _run_collaborative_discussion(self, agent_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Run collaborative discussion between agents.
        
        Args:
            agent_analyses: Initial analyses from each agent
            
        Returns:
            Dictionary mapping agent roles to their final decisions
        """
        agent_decisions = {}
        
        # For each agent, have them review other agents' analyses
        for role, agent in self.agents.items():
            self.logger.logger.info(f"Running collaborative discussion for {role}")
            
            # Collect other agents' analyses
            other_analyses = {}
            for other_role, analysis_data in agent_analyses.items():
                if other_role != role:
                    other_analyses[other_role] = analysis_data["analysis"]
            
            # Create collaborative prompt
            if len(other_analyses) > 0:
                other_analyses_text = "\n\n".join([f"{other_role}:\n{analysis}" 
                                                 for other_role, analysis in other_analyses.items()])
                        
                collaborative_prompt = DISCUSSION_PROMPTS["collaborative_discussion"].format(
                    initial_analysis=agent_analyses[role]['analysis'],
                    teammates_analyses=other_analyses_text
                )

                # Apply mutual monitoring if enabled
                if self.use_mutual_monitoring and self.mutual_monitor:
                    # For each teammate's analysis, generate monitoring feedback
                    for other_role, analysis in other_analyses.items():
                        # Extract the response
                        other_agent = self.agents[other_role]
                        extract = other_agent.extract_response(analysis)
                        
                        # Monitor the response
                        monitoring_result = self.mutual_monitor.monitor_agent_response(
                            other_role,
                            analysis,
                            None  # No separate reasoning provided
                        )
                        
                        # Generate feedback if issues detected
                        if monitoring_result["issues_detected"]:
                            feedback = self.mutual_monitor.generate_feedback(
                                monitoring_result,
                                role
                            )
                            
                            # Log to monitoring channel
                            self.logger.log_monitoring_action(
                                role,
                                other_role,
                                monitoring_result["issues"],
                                feedback
                            )
                            
                            # Add monitoring feedback to collaborative prompt
                            collaborative_prompt += f"""
                            
                            Based on your monitoring of {other_role}'s analysis, you've identified these issues:
                            {feedback}
                            
                            Consider these points in your final response.
                            """
                
                # Enhance with shared mental model if enabled
                if self.use_shared_mental_model and self.mental_model:
                    collaborative_prompt = self.mental_model.enhance_agent_prompt(role, collaborative_prompt)
                
                # Use closed-loop communication if enabled
                if self.use_closed_loop_comm and self.comm_handler:
                    # Select first agent that isn't the current one to acknowledge
                    responder_role = next((r for r in self.agents if r != role), None)
                    
                    if responder_role:
                        responder = self.agents[responder_role]
                        
                        # Facilitate closed-loop exchange
                        exchange = self.comm_handler.facilitate_exchange(
                            agent,
                            responder,
                            collaborative_prompt
                        )
                        
                        # Log to closed-loop channel
                        self.logger.log_closed_loop(
                            "collaborative_discussion",
                            role,
                            responder_role,
                            exchange[0],
                            exchange[1],
                            exchange[2]
                        )
                        
                        # Extract the substantive content
                        content = self.comm_handler.extract_content_from_exchange(exchange)
                        
                        # Store the final decision
                        final_decision = content["sender_content"]
                        
                        # Store in results
                        self.results["exchanges"].append({
                            "type": "collaborative_discussion",
                            "communication": "closed_loop",
                            "sender": role,
                            "receiver": responder_role,
                            "initial_message": exchange[0],
                            "acknowledgment": exchange[1],
                            "verification": exchange[2]
                        })
                    else:
                        # Fallback to standard communication
                        final_decision = agent.chat(collaborative_prompt)
                else:
                    # Standard communication
                    final_decision = agent.chat(collaborative_prompt)
                    
                    # Log to main discussion channel
                    self.logger.log_main_discussion(
                        "collaborative_discussion",
                        role,
                        final_decision
                    )
                    
                    # Store in results
                    self.results["exchanges"].append({
                        "type": "collaborative_discussion",
                        "communication": "standard",
                        "sender": role,
                        "message": final_decision
                    })
            else:
                # If there are no other agents, use the initial analysis
                final_decision = agent_analyses[role]["analysis"]
            
            # Update shared mental model if enabled
            if self.use_shared_mental_model and self.mental_model:
                understanding = self.mental_model.extract_understanding_from_message(final_decision)
                self.mental_model.update_shared_understanding(role, understanding)
                
                # Log to shared mental model channel
                self.logger.log_mental_model_update(
                    role,
                    understanding,
                    self.mental_model.convergence_metrics[-1]["overall_convergence"] if self.mental_model.convergence_metrics else 0.0
                )
            
            # Extract the response structure
            extracted = agent.extract_response(final_decision)
            
            # Get agent weight from knowledge base
            weight = agent.get_from_knowledge_base("weight")
            if weight is None:
                weight = 0.2  # Default weight
            
            # Store the agent's decision with weight
            agent_decisions[role] = {
                "final_decision": final_decision,
                "extract": extracted,
                "weight": weight  # Include the weight
            }
        
        # If leadership is enabled, have leader synthesize the team's decision
        if self.use_team_leadership and self.leader:
            # Create synthesis context with all agent decisions
            context = "\n\n".join([f"{role}:\n{decision['final_decision']}" 
                                 for role, decision in agent_decisions.items() 
                                 if role != self.leader.role])
            
            # Have leader synthesize
            leader_synthesis = self.leader.leadership_action("synthesize", context)
            
            # Log to leadership channel
            self.logger.log_leadership_action(
                "synthesis",
                leader_synthesis
            )
            
            # Log to main discussion channel
            self.logger.log_main_discussion(
                "leadership_synthesis",
                self.leader.role,
                leader_synthesis
            )
            
            # Store in results
            self.results["exchanges"].append({
                "type": "leadership_synthesis",
                "communication": "standard",
                "sender": self.leader.role,
                "message": leader_synthesis
            })
            
            # Extract the response
            leader_extract = self.leader.extract_response(leader_synthesis)
            
            # Update leader's decision
            agent_decisions[self.leader.role] = {
                "final_decision": leader_synthesis,
                "extract": leader_extract
            }
        
        return agent_decisions
    


    def _apply_decision_methods(self, agent_decisions):
        """
        Apply appropriate decision methods based on task level.
        
        Args:
            agent_decisions: Dictionary of agent decisions
            
        Returns:
            Dictionary with decision results
        """
        # Determine if this is an MDT (advanced) task
        is_mdt_task = False
        for agent_role in agent_decisions.keys():
            if any(prefix in agent_role for prefix in ["1_", "2_", "3_"]):
                is_mdt_task = True
                break
        
        if is_mdt_task:
            logging.info("Using MDT decision process for advanced query")
            return self._apply_mdt_decision_methods(agent_decisions)
        else:
            # Basic or intermediate level - apply standard decision methods
            return self._apply_standard_decision_methods(agent_decisions)
    

    def _apply_standard_decision_methods(self, agent_decisions):
        """Apply standard decision methods for basic/intermediate tasks."""
        # Apply majority voting
        majority_result = self.decision_methods.majority_voting(agent_decisions)
        logging.info(f"Majority voting result: {majority_result}")
        
        # Apply weighted voting
        weighted_result = self.decision_methods.weighted_voting(agent_decisions)
        logging.info(f"Weighted voting result: {weighted_result}")
        
        # Apply Borda count
        borda_result = self.decision_methods.borda_count(agent_decisions)
        logging.info(f"Borda count result: {borda_result}")
        
        return {
            "majority_voting": majority_result,
            "weighted_voting": weighted_result,
            "borda_count": borda_result
        }


    def _apply_mdt_decision_methods(self, agent_decisions):
        """Apply decision methods for MDT (advanced) tasks."""
        # Group agents by team
        team_decisions = {}
        for agent_role, response in agent_decisions.items():
            parts = agent_role.split("_")
            if len(parts) >= 2:
                team_name = "_".join(parts[:2])
                if team_name not in team_decisions:
                    team_decisions[team_name] = {}
                team_decisions[team_name][agent_role] = response
        
        # Make team-level decisions
        team_results = {}
        for team_name, team_responses in team_decisions.items():
            team_results[team_name] = self.decision_methods.majority_voting(team_responses)
        
        # Identify final team
        final_team_name = None
        for team_name in sorted(team_decisions.keys(), reverse=True):
            if team_name.startswith("3_") or "Final" in team_name:
                final_team_name = team_name
                break
        
        # Use final team's decisions
        final_agent_responses = {}
        if final_team_name:
            final_agent_responses = team_decisions[final_team_name]
        else:
            # Use all responses if no final team found
            final_agent_responses = agent_decisions
        
        # Apply decision methods to final team
        majority_result = self.decision_methods.majority_voting(final_agent_responses)
        logging.info(f"MDT Majority voting result: {majority_result}")
        
        weighted_result = self.decision_methods.weighted_voting(final_agent_responses)
        logging.info(f"MDT Weighted voting result: {weighted_result}")
        
        borda_result = self.decision_methods.borda_count(final_agent_responses)
        logging.info(f"MDT Borda count result: {borda_result}")
        
        return {
            "majority_voting": majority_result,
            "weighted_voting": weighted_result,
            "borda_count": borda_result,
            "mdt_process": {
                "team_decisions": team_results,
                "decision_team": final_team_name
            }
        }


    def save_results(self) -> str:
        """
        Save simulation results to file.
        
        Returns:
            Path to the saved results file
        """
        output_path = os.path.join(config.OUTPUT_DIR, f"{self.simulation_id}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.logger.info(f"Results saved to {output_path}")
        return output_path
    

    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate the performance of the simulation.
        
        Returns:
            Dictionary with performance metrics
        """
        performance = {
            "task_performance": self._evaluate_task_performance(),
            "teamwork_performance": self._evaluate_teamwork_performance()
        }
        
        return performance
    

    def _evaluate_task_performance(self) -> Dict[str, Any]:
        """
        Evaluate performance on the task itself.
        
        Returns:
            Dictionary with task performance metrics
        """
        task_type = config.TASK["type"]
        
        if task_type == "ranking":
            return self._evaluate_ranking_performance()
        elif task_type == "mcq":
            return self._evaluate_mcq_performance()
        else:
            return {"metric": "qualitative", "note": "No quantitative metric for this task type"}
    

    def _evaluate_ranking_performance(self) -> Dict[str, Any]:
        """Evaluate performance on ranking tasks."""
        ground_truth = config.TASK.get("ground_truth", [])
        
        if not ground_truth:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        # Evaluate each decision method
        for method, result in self.results["decision_results"].items():
            if result and "final_ranking" in result:
                ranking = result["final_ranking"]
                
                # Calculate correlation with ground truth
                correlation = self._calculate_rank_correlation(ranking, ground_truth)
                
                # Calculate error (sum of absolute differences in position)
                error = self._calculate_ranking_error(ranking, ground_truth)
                
                metrics[method] = {
                    "correlation": correlation,
                    "error": error
                }
        
        return metrics
    

    def _evaluate_mcq_performance(self) -> Dict[str, Any]:
        """Evaluate performance on MCQ tasks."""
        ground_truth = config.TASK.get("ground_truth")
        
        if not ground_truth:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        # Evaluate each decision method
        for method, result in self.results["decision_results"].items():
            if result and "winning_option" in result:
                selected = result["winning_option"]
                
                # Check if the selected option matches ground truth
                correct = selected == ground_truth
                
                metrics[method] = {
                    "correct": correct,
                    "confidence": result.get("confidence", 0)
                }
        
        return metrics
    

    def _evaluate_teamwork_performance(self) -> Dict[str, Any]:
        """
        Evaluate performance of the teamwork components.
        
        Returns:
            Dictionary with teamwork performance metrics
        """
        teamwork_metrics = {}
        
        # Closed-loop communication metrics
        if "closed_loop_communication" in self.results["teamwork_metrics"]:
            comm_metrics = self.results["teamwork_metrics"]["closed_loop_communication"]
            teamwork_metrics["closed_loop_communication"] = {
                "effectiveness": comm_metrics.get("effectiveness_rating", "N/A"),
                "misunderstanding_rate": comm_metrics.get("misunderstanding_rate", 0),
                "total_exchanges": comm_metrics.get("total_exchanges", 0)
            }
        
        # Mutual monitoring metrics
        if "mutual_monitoring" in self.results["teamwork_metrics"]:
            monitor_metrics = self.results["teamwork_metrics"]["mutual_monitoring"]
            teamwork_metrics["mutual_monitoring"] = {
                "effectiveness": monitor_metrics.get("team_monitoring_effectiveness", "N/A"),
                "issue_resolution_rate": monitor_metrics.get("issue_resolution_rate", 0),
                "total_issues_detected": monitor_metrics.get("total_issues_detected", 0)
            }
        
        # Shared mental model metrics
        if "shared_mental_model" in self.results["teamwork_metrics"]:
            model_metrics = self.results["teamwork_metrics"]["shared_mental_model"]
            teamwork_metrics["shared_mental_model"] = {
                "effectiveness": model_metrics.get("effectiveness_rating", "N/A"),
                "convergence_trend": model_metrics.get("convergence_trend", "unknown"),
                "final_convergence": model_metrics.get("final_convergence", 0)
            }
        
        # Leadership metrics
        if self.use_team_leadership:
            # Count leadership actions
            leadership_actions = sum(1 for exchange in self.results["exchanges"] 
                                  if exchange.get("type") in ["leadership_definition", "leadership_synthesis"])
            
            teamwork_metrics["leadership"] = {
                "leader_role": self.leader.role if self.leader else "None",
                "leadership_actions": leadership_actions
            }
        
        return teamwork_metrics
    

    def _calculate_rank_correlation(self, ranking1: List[str], ranking2: List[str]) -> float:
        """
        Calculate Spearman's rank correlation between two rankings.
        
        Args:
            ranking1: First ranking list
            ranking2: Second ranking list
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Convert to position dictionaries
        pos1 = {item: i for i, item in enumerate(ranking1)}
        pos2 = {item: i for i, item in enumerate(ranking2)}
        
        # Get common items
        common_items = set(ranking1) & set(ranking2)
        
        if not common_items:
            return 0.0
        
        # Calculate differences
        d_squared_sum = sum((pos1[item] - pos2[item])**2 for item in common_items)
        n = len(common_items)
        
        # Spearman's formula
        correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        
        return correlation
    

    def _calculate_ranking_error(self, ranking1: List[str], ranking2: List[str]) -> int:
        """
        Calculate error between two rankings (sum of absolute position differences).
        
        Args:
            ranking1: First ranking list
            ranking2: Second ranking list
            
        Returns:
            Sum of absolute differences in position
        """
        # Convert to position dictionaries
        pos1 = {item: i for i, item in enumerate(ranking1)}
        pos2 = {item: i for i, item in enumerate(ranking2)}
        
        # Calculate error
        error = 0
        for item in set(pos1.keys()) & set(pos2.keys()):
            error += abs(pos1[item] - pos2[item])
        
        return error