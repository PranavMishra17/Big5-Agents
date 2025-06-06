"""
Enhanced simulator.py with parallel processing and timeout handling.
"""

import os
import logging
import json
import re
import traceback
import concurrent.futures
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from components.modular_agent import ModularAgent, create_agent_team
from components.agent_recruitment import determine_complexity, recruit_agents
from components.closed_loop import ClosedLoopCommunication
from components.mutual_monitoring import MutualMonitoring
from components.shared_mental_model import SharedMentalModel
from components.decision_methods import DecisionMethods
from utils.logger import SimulationLogger
import config
from components.team_orientation import TeamOrientation
from components.mutual_trust import MutualTrust

from utils.prompts import DISCUSSION_PROMPTS, LEADERSHIP_PROMPTS, get_adaptive_prompt


class AgentSystemSimulator:
    """
    Enhanced simulator with parallel processing and timeout handling.
    """
    
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
             recruitment_pool: str = None,
             n_max: int = 5):
        """Initialize the simulator with parallel processing support."""
        
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
        self.n_max = n_max if n_max is not None else 5
        
        self.metadata = {}
        self.evaluation_data = getattr(config, 'TASK_EVALUATION', {})
        
        # Parallel processing settings
        self.enable_parallel = config.ENABLE_PARALLEL_PROCESSING and len(config.AZURE_DEPLOYMENTS) > 1
        self.max_workers = config.MAX_PARALLEL_WORKERS if self.enable_parallel else 1
        
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
            "n_max": self.n_max,
            "task": config.TASK.get("name", "Unknown"),
            "parallel_processing": self.enable_parallel,
            "max_workers": self.max_workers
        }
        
        # Setup logging
        self.logger = SimulationLogger(
            simulation_id=self.simulation_id,
            log_dir=config.LOG_DIR,
            config=self.config
        )

        # Create agent team
        self._create_agent_team()
        
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
            self.mental_model.initialize_task_model(config.TASK)
            self.mental_model.initialize_team_model(list(self.agents.keys()))
        
        # Store results
        self.results = {
            "simulation_id": self.simulation_id,
            "config": self.config,
            "exchanges": [],
            "decision_results": {}
        }
        
        self.logger.logger.info(f"Initialized simulation {self.simulation_id} with parallel processing: {self.enable_parallel}")

    def _create_agent_team(self):
        """Create agent team with proper recruitment handling."""
        if self.use_recruitment and config.TASK.get("description"):
            try:
                from components.agent_recruitment import determine_complexity, recruit_agents
                complexity = determine_complexity(config.TASK["description"], self.recruitment_method)
                self.metadata["complexity"] = complexity
                
                agents, leader = recruit_agents(
                    config.TASK["description"],
                    complexity,
                    self.recruitment_pool,
                    self.n_max,
                    self.recruitment_method
                )
                self.agents = agents
                self.leader = leader
                
            except Exception as e:
                logging.error(f"Recruitment failed: {str(e)}, using default team")
                from components.modular_agent import create_agent_team
                self.agents, self.leader = create_agent_team(
                    use_team_leadership=self.use_team_leadership,
                    use_closed_loop_comm=self.use_closed_loop_comm,
                    use_mutual_monitoring=self.use_mutual_monitoring,
                    use_shared_mental_model=self.use_shared_mental_model,
                    use_team_orientation=self.use_team_orientation,
                    use_mutual_trust=self.use_mutual_trust,
                    random_leader=self.random_leader,
                    use_recruitment=False,
                    n_max=self.n_max
                )
        else:
            from components.modular_agent import create_agent_team
            self.agents, self.leader = create_agent_team(
                use_team_leadership=self.use_team_leadership,
                use_closed_loop_comm=self.use_closed_loop_comm,
                use_mutual_monitoring=self.use_mutual_monitoring,
                use_shared_mental_model=self.use_shared_mental_model,
                use_team_orientation=self.use_team_orientation,
                use_mutual_trust=self.use_mutual_trust,
                random_leader=self.random_leader,
                use_recruitment=False,
                n_max=self.n_max
            )

    def _execute_agent_task_parallel(self, agent_role_pairs: List[Tuple[str, Any, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Execute agent tasks in parallel with timeout handling.
        
        Args:
            agent_role_pairs: List of (role, agent, task_type) tuples
            
        Returns:
            Dictionary mapping roles to their results
        """
        results = {}
        failed_agents = []
        
        if not self.enable_parallel or len(agent_role_pairs) == 1:
            # Sequential execution for single agent or when parallel disabled
            for role, agent, task_type in agent_role_pairs:
                try:
                    if task_type == "analyze":
                        result = agent.analyze_task()
                    elif task_type == "final_decision":
                        result = agent  # This will be handled in the calling function
                    else:
                        result = agent.chat(task_type)  # task_type is the prompt
                    
                    results[role] = {
                        "analysis" if task_type == "analyze" else "result": result,
                        "extract": agent.extract_response(result) if task_type == "analyze" else None
                    }
                    
                except Exception as e:
                    self.logger.logger.error(f"Error in sequential execution for {role}: {str(e)}")
                    failed_agents.append((role, str(e)))
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_role = {}
                
                for role, agent, task_type in agent_role_pairs:
                    if task_type == "analyze":
                        future = executor.submit(self._safe_agent_analyze, agent, role)
                    elif task_type == "final_decision":
                        future = executor.submit(self._safe_agent_final_decision, agent, role, task_type)
                    else:
                        future = executor.submit(self._safe_agent_chat, agent, role, task_type)
                    
                    future_to_role[future] = role
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(future_to_role, timeout=config.INACTIVITY_TIMEOUT * 2):
                    role = future_to_role[future]
                    try:
                        result = future.result(timeout=config.INACTIVITY_TIMEOUT)
                        results[role] = result
                        self.logger.logger.info(f"Parallel task completed for {role}")
                        
                    except concurrent.futures.TimeoutError:
                        self.logger.logger.error(f"Timeout for {role} in parallel execution")
                        failed_agents.append((role, "Timeout"))
                    except Exception as e:
                        self.logger.logger.error(f"Error for {role} in parallel execution: {str(e)}")
                        failed_agents.append((role, str(e)))
        
        # Handle failed agents with retry
        if failed_agents:
            self.logger.logger.warning(f"Retrying {len(failed_agents)} failed agents sequentially")
            for role, error in failed_agents:
                try:
                    # Find the agent and task type
                    for r, agent, task_type in agent_role_pairs:
                        if r == role:
                            self.logger.logger.info(f"Retrying {role} with task type {task_type}")
                            
                            if task_type == "analyze":
                                result = agent.analyze_task()
                                results[role] = {
                                    "analysis": result,
                                    "extract": agent.extract_response(result)
                                }
                            elif task_type == "final_decision":
                                # This needs to be handled in the calling function
                                results[role] = {"needs_retry": True, "agent": agent}
                            else:
                                result = agent.chat(task_type)
                                results[role] = {"result": result}
                            break
                            
                except Exception as e:
                    self.logger.logger.error(f"Retry also failed for {role}: {str(e)}")
                    results[role] = {"error": str(e)}
        
        return results

    def _safe_agent_analyze(self, agent: Any, role: str) -> Dict[str, Any]:
        """Safely execute agent analysis with error handling."""
        try:
            analysis = agent.analyze_task()
            return {
                "analysis": analysis,
                "extract": agent.extract_response(analysis)
            }
        except Exception as e:
            self.logger.logger.error(f"Error in agent analysis for {role}: {str(e)}")
            raise

    def _safe_agent_chat(self, agent: Any, role: str, prompt: str) -> Dict[str, Any]:
        """Safely execute agent chat with error handling."""
        try:
            result = agent.chat(prompt)
            return {"result": result}
        except Exception as e:
            self.logger.logger.error(f"Error in agent chat for {role}: {str(e)}")
            raise

    def _safe_agent_final_decision(self, agent: Any, role: str, prompt: str) -> Dict[str, Any]:
        """Safely execute agent final decision with error handling."""
        try:
            decision = agent.chat(prompt)
            return {
                "final_decision": decision,
                "extract": agent.extract_response(decision),
                "weight": agent.get_from_knowledge_base("weight") or 0.2
            }
        except Exception as e:
            self.logger.logger.error(f"Error in agent final decision for {role}: {str(e)}")
            raise

    def run_simulation(self):
        """
        Run the enhanced 3-round simulation process with parallel processing.
        
        Returns:
            Dictionary with simulation results
        """
        # ROUND 1: Independent Analysis (Parallel)
        self.logger.logger.info("ROUND 1: Independent task analysis (parallel execution)")
        round1_analyses = self._run_round1_independent_analysis()
        
        # Leadership definition if enabled (between rounds)
        if self.use_team_leadership and self.leader:
            self.logger.logger.info("Leadership phase: Defining task approach")
            self._run_leadership_definition()
        
        # ROUND 2: Collaborative Discussion (Sequential for interaction dependencies)
        self.logger.logger.info("ROUND 2: Collaborative discussion")
        round2_discussions = self._run_round2_collaborative_discussion(round1_analyses)
        
        # ROUND 3: Final Independent Decision (Parallel)
        self.logger.logger.info("ROUND 3: Final independent decisions (parallel execution)")
        round3_decisions = self._run_round3_final_decisions(round1_analyses, round2_discussions)
        
        # Apply decision methods to final decisions
        self.logger.logger.info("Applying decision methods to final decisions")
        decision_results = self._apply_decision_methods(round3_decisions)
        
        # Store results
        self.results["decision_results"] = decision_results
        self.results["round1_analyses"] = round1_analyses
        self.results["round2_discussions"] = round2_discussions
        self.results["round3_decisions"] = round3_decisions
        
        # Add teamwork metrics
        self._collect_teamwork_metrics()
        
        # Save results
        self.save_results()
        
        # Return enhanced results structure
        return {
            "simulation_metadata": {
                "simulation_id": self.simulation_id,
                "timestamp": datetime.now().isoformat(),
                "parallel_processing": self.enable_parallel,
                "task_info": {
                    "name": config.TASK.get("name", ""),
                    "type": config.TASK.get("type", ""),
                    "description": config.TASK.get("description", "")[:200] + "...",
                    "options": config.TASK.get("options", [])
                }
            },
            "agent_analyses": round1_analyses,      # Round 1 independent analyses
            "agent_responses": round3_decisions,    # Round 3 final decisions
            "exchanges": self.results.get("exchanges", []),
            "decision_results": decision_results
        }

    def _run_round1_independent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        ROUND 1: Each agent analyzes the task independently (PARALLEL EXECUTION).
        
        Returns:
            Dictionary mapping agent roles to their independent analyses
        """
        # Prepare agent tasks for parallel execution
        agent_tasks = [(role, agent, "analyze") for role, agent in self.agents.items()]
        
        # Execute in parallel
        parallel_results = self._execute_agent_task_parallel(agent_tasks)
        
        agent_analyses = {}
        
        for role, result in parallel_results.items():
            if "error" in result:
                self.logger.logger.error(f"Failed to get analysis from {role}: {result['error']}")
                agent_analyses[role] = {
                    "analysis": f"Error occurred: {result['error']}",
                    "extract": {"error": result['error']}
                }
            else:
                analysis = result["analysis"]
                
                # Log to main discussion channel
                self.logger.log_main_discussion(
                    "round1_independent_analysis",
                    role,
                    analysis
                )
                
                # Store analysis
                agent_analyses[role] = {
                    "analysis": analysis,
                    "extract": result["extract"]
                }
                
                # Update shared mental model if enabled (but don't share between agents yet)
                if self.use_shared_mental_model and self.mental_model:
                    understanding = self.mental_model.extract_understanding_from_message(analysis)
                    self.mental_model.update_shared_understanding(role, understanding)
        
        self.logger.logger.info(f"Round 1 completed: {len(agent_analyses)} analyses collected")
        return agent_analyses

    def _run_round2_collaborative_discussion(self, round1_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        ROUND 2: Agents discuss based on sanitized peer analyses (SEQUENTIAL for dependencies).
        
        Args:
            round1_analyses: Results from Round 1
            
        Returns:
            Dictionary mapping agent roles to their discussion contributions
        """
        round2_discussions = {}
        
        # Sequential execution to maintain discussion flow
        for role, agent in self.agents.items():
            self.logger.logger.info(f"Round 2: Collaborative discussion for {role}")
            
            # Collect sanitized analyses from other agents
            other_analyses = {}
            for other_role, analysis_data in round1_analyses.items():
                if other_role != role:
                    other_analyses[other_role] = analysis_data["analysis"]
            
            if len(other_analyses) > 0:
                # Create discussion prompt with peer analyses
                other_analyses_text = "\n\n".join([f"{other_role}:\n{analysis}" 
                                                for other_role, analysis in other_analyses.items()])
                
                discussion_prompt = f"""
                You have completed your initial analysis of the task. Now you can see the reasoning and analysis from your teammates (their final answers have been removed to avoid bias).
                
                Your initial analysis:
                {round1_analyses[role]['analysis']}
                
                Your teammates' reasoning:
                {other_analyses_text}
                
                Based on these different perspectives:
                1. Identify points where you agree or disagree with your teammates
                2. Question any reasoning that seems unclear or potentially flawed
                3. Share additional insights that might help the team
                4. Discuss any concerns or alternative approaches you see
                
                DO NOT provide a final answer in this round. Focus on discussion and analysis only.
                This is a collaborative discussion to better understand the problem before making your final decision.
                """
                
                # Apply teamwork components to the discussion
                if self.use_mutual_monitoring and self.mutual_monitor:
                    # Monitor peer analyses for issues
                    for other_role, analysis in other_analyses.items():
                        other_agent = self.agents[other_role]
                        extract = other_agent.extract_response(analysis)
                        
                        monitoring_result = self.mutual_monitor.monitor_agent_response(
                            other_role, analysis, None
                        )
                        
                        if monitoring_result["issues_detected"]:
                            feedback = self.mutual_monitor.generate_feedback(
                                monitoring_result, role
                            )
                            discussion_prompt += f"""
                            
                            Based on your monitoring of {other_role}'s analysis, you've identified these issues:
                            {feedback}
                            
                            Consider these points in your discussion.
                            """
                
                # Get discussion response (with timeout handling)
                try:
                    discussion_response = agent.chat(discussion_prompt)
                    
                    # Log to main discussion channel
                    self.logger.log_main_discussion(
                        "round2_collaborative_discussion",
                        role,
                        discussion_response
                    )
                    
                    # Store in results
                    self.results["exchanges"].append({
                        "type": "round2_collaborative_discussion",
                        "communication": "standard",
                        "sender": role,
                        "message": discussion_response
                    })
                    
                    round2_discussions[role] = discussion_response
                    
                except Exception as e:
                    self.logger.logger.error(f"Error in Round 2 discussion for {role}: {str(e)}")
                    round2_discussions[role] = f"Error in discussion: {str(e)}"
                    
            else:
                # Single agent case
                round2_discussions[role] = "No teammates to discuss with."
        
        return round2_discussions

    def _run_round3_final_decisions(self, round1_analyses: Dict[str, Dict[str, Any]], 
                                   round2_discussions: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        ROUND 3: Each agent makes final independent decision (PARALLEL EXECUTION).
        
        Args:
            round1_analyses: Results from Round 1
            round2_discussions: Results from Round 2
            
        Returns:
            Dictionary mapping agent roles to their final decisions
        """
        # Prepare final decision prompts for parallel execution
        agent_tasks = []
        
        for role, agent in self.agents.items():
            task_type = config.TASK.get("type", "mcq")
            
            try:
                final_prompt = get_adaptive_prompt(
                    "final_decision",
                    task_type,
                    initial_analysis=round1_analyses[role]['analysis'],
                    discussion_summary=round2_discussions.get(role, "No discussion occurred.")
                )
            except Exception as e:
                # Fallback final decision prompt
                if task_type == "multi_choice_mcq":
                    final_prompt = f"""
                    Based on your initial analysis and the team discussion, provide your final answer to this multi-choice question.
                    
                    Your initial analysis:
                    {round1_analyses[role]['analysis']}
                    
                    Team discussion insights:
                    {round2_discussions.get(role, "No discussion occurred.")}
                    
                    Now provide your final answer. Remember: This is a multi-choice question where multiple answers may be correct.
                    Begin with "ANSWERS: X,Y,Z" (replace with ALL correct option letters).
                    Then provide your final reasoning.
                    """
                elif task_type == "yes_no_maybe":
                    final_prompt = f"""
                    Based on your initial analysis and the team discussion, provide your final answer to this research question.
                    
                    Your initial analysis:
                    {round1_analyses[role]['analysis']}
                    
                    Team discussion insights:
                    {round2_discussions.get(role, "No discussion occurred.")}
                    
                    Now provide your final answer.
                    Begin with "ANSWER: X" (replace X with yes, no, or maybe).
                    Then provide your final scientific reasoning.
                    """
                else:
                    final_prompt = f"""
                    Based on your initial analysis and the team discussion, provide your final answer.
                    
                    Your initial analysis:
                    {round1_analyses[role]['analysis']}
                    
                    Team discussion insights:
                    {round2_discussions.get(role, "No discussion occurred.")}
                    
                    Now provide your final answer.
                    Begin with "ANSWER: X" (replace X with your chosen option A, B, C, or D).
                    Then provide your final reasoning.
                    """
            
            agent_tasks.append((role, agent, final_prompt))
        
        # Execute final decisions in parallel
        parallel_results = self._execute_agent_task_parallel(agent_tasks)
        
        round3_decisions = {}
        
        for role, result in parallel_results.items():
            if "error" in result:
                self.logger.logger.error(f"Failed to get final decision from {role}: {result['error']}")
                # Create a fallback decision
                round3_decisions[role] = {
                    "final_decision": f"Error occurred: {result['error']}",
                    "extract": {"error": result['error']},
                    "weight": 0.2
                }
            elif "needs_retry" in result:
                # Handle retry for final decision
                try:
                    agent = result["agent"]
                    # Find the prompt for this agent
                    final_prompt = None
                    for r, a, prompt in agent_tasks:
                        if r == role:
                            final_prompt = prompt
                            break
                    
                    if final_prompt:
                        final_decision = agent.chat(final_prompt)
                        extracted = agent.extract_response(final_decision)
                        weight = agent.get_from_knowledge_base("weight") or 0.2
                        
                        round3_decisions[role] = {
                            "final_decision": final_decision,
                            "extract": extracted,
                            "weight": weight
                        }
                    else:
                        raise Exception("Could not find prompt for retry")
                        
                except Exception as e:
                    self.logger.logger.error(f"Retry failed for {role}: {str(e)}")
                    round3_decisions[role] = {
                        "final_decision": f"Retry failed: {str(e)}",
                        "extract": {"error": str(e)},
                        "weight": 0.2
                    }
            else:
                final_decision = result["result"]
                
                # Log to main discussion channel
                self.logger.log_main_discussion(
                    "round3_final_decision",
                    role,
                    final_decision
                )
                
                # Store in results
                self.results["exchanges"].append({
                    "type": "round3_final_decision",
                    "communication": "standard",
                    "sender": role,
                    "message": final_decision
                })
                
                # Extract the response structure
                agent = self.agents[role]
                extracted = agent.extract_response(final_decision)
                
                # Get agent weight
                weight = agent.get_from_knowledge_base("weight") or 0.2
                
                # Store the agent's final decision
                round3_decisions[role] = {
                    "final_decision": final_decision,
                    "extract": extracted,
                    "weight": weight
                }
        
        # Leadership synthesis if enabled (based on Round 3 decisions)
        if self.use_team_leadership and self.leader:
            context = "\n\n".join([f"{role}:\n{decision['final_decision']}" 
                                for role, decision in round3_decisions.items() 
                                if role != self.leader.role])
            
            leader_synthesis = self.leader.leadership_action("synthesize", context)
            
            self.logger.log_leadership_action("synthesis", leader_synthesis)
            self.logger.log_main_discussion("leadership_synthesis", self.leader.role, leader_synthesis)
            
            self.results["exchanges"].append({
                "type": "leadership_synthesis",
                "communication": "standard",
                "sender": self.leader.role,
                "message": leader_synthesis
            })
            
            # Update leader's decision
            leader_extract = self.leader.extract_response(leader_synthesis)
            round3_decisions[self.leader.role] = {
                "final_decision": leader_synthesis,
                "extract": leader_extract,
                "weight": round3_decisions[self.leader.role].get("weight", 0.2)
            }
        
        self.logger.logger.info(f"Round 3 completed: {len(round3_decisions)} decisions collected")
        return round3_decisions

    def _run_leadership_definition(self) -> str:
        """Have the leader define the team's approach (between Round 1 and 2)."""
        if not self.leader:
            return "No leader designated for this simulation."
        
        leadership_prompt = f"""
        As the leader, define the team's overall approach to solving this task:
        
        {config.TASK['description']}
        
        Please:
        1. Break down this task into clear steps
        2. Define how the team should work together
        3. Specify what each team member should contribute based on their expertise
        4. Outline how we will reach a final decision
        
        Provide clear, specific guidance that will help the team work effectively together.
        """
        
        leader_definition = self.leader.leadership_action("define_task")
        
        self.logger.log_main_discussion("leadership_definition", self.leader.role, leader_definition)
        self.logger.log_leadership_action("task_definition", leader_definition)
        
        self.results["exchanges"].append({
            "type": "leadership_definition",
            "communication": "standard",
            "sender": self.leader.role,
            "message": leader_definition
        })
        
        return leader_definition

    def _apply_decision_methods(self, agent_decisions):
        """Apply decision methods to agent decisions."""
        # Determine if this is an MDT (advanced) task
        is_mdt_task = any(any(prefix in agent_role for prefix in ["1_", "2_", "3_"]) 
                         for agent_role in agent_decisions.keys())
        
        if is_mdt_task:
            logging.info("Using MDT decision process for advanced query")
            return self._apply_mdt_decision_methods(agent_decisions)
        else:
            return self._apply_standard_decision_methods(agent_decisions)

    def _apply_standard_decision_methods(self, agent_decisions):
        """Apply standard decision methods."""
        logging.info(f"Applying standard decision methods to {len(agent_decisions)} agent responses")
        
        majority_result = self.decision_methods.majority_voting(agent_decisions)
        weighted_result = self.decision_methods.weighted_voting(agent_decisions)
        borda_result = self.decision_methods.borda_count(agent_decisions)
        
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
        
        # Use final team's decisions
        final_team_name = None
        for team_name in sorted(team_decisions.keys(), reverse=True):
            if team_name.startswith("3_") or "Final" in team_name:
                final_team_name = team_name
                break
        
        final_agent_responses = team_decisions.get(final_team_name, agent_decisions)
        
        majority_result = self.decision_methods.majority_voting(final_agent_responses)
        weighted_result = self.decision_methods.weighted_voting(final_agent_responses)
        borda_result = self.decision_methods.borda_count(final_agent_responses)
        
        return {
            "majority_voting": majority_result,
            "weighted_voting": weighted_result,
            "borda_count": borda_result,
            "mdt_process": {
                "team_decisions": team_results,
                "decision_team": final_team_name
            }
        }

    def _collect_teamwork_metrics(self):
        """Collect teamwork metrics from enabled components."""
        self.results["teamwork_metrics"] = {}
        
        if self.use_closed_loop_comm and self.comm_handler:
            self.results["teamwork_metrics"]["closed_loop_communication"] = self.comm_handler.get_communication_metrics()
            
        if self.use_mutual_monitoring and self.mutual_monitor:
            self.results["teamwork_metrics"]["mutual_monitoring"] = self.mutual_monitor.analyze_team_performance()
            
        if self.use_shared_mental_model and self.mental_model:
            self.results["teamwork_metrics"]["shared_mental_model"] = self.mental_model.analyze_mental_model_effectiveness()
        
        if self.use_team_orientation and self.team_orientation:
            self.results["teamwork_metrics"]["team_orientation"] = self.team_orientation.get_team_orientation_metrics()
            
        if self.use_mutual_trust and self.mutual_trust:
            self.results["teamwork_metrics"]["mutual_trust"] = self.mutual_trust.get_trust_metrics()

    def save_results(self) -> str:
        """Save simulation results to file."""
        output_path = os.path.join(config.OUTPUT_DIR, f"{self.simulation_id}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.logger.info(f"Results saved to {output_path}")
        return output_path

    def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate the performance of the simulation."""
        performance = {
            "task_performance": self._evaluate_task_performance(),
            "teamwork_performance": self._evaluate_teamwork_performance()
        }
        return performance

    def _evaluate_task_performance(self) -> Dict[str, Any]:
        """Evaluate performance on the task itself."""
        task_type = config.TASK["type"]
        
        if task_type == "ranking":
            return self._evaluate_ranking_performance()
        elif task_type == "mcq":
            return self._evaluate_mcq_performance()
        elif task_type == "multi_choice_mcq":
            return self._evaluate_multi_choice_performance()
        elif task_type == "yes_no_maybe":
            return self._evaluate_yes_no_maybe_performance()
        else:
            return {"metric": "qualitative", "note": "No quantitative metric for this task type"}

    def _evaluate_mcq_performance(self) -> Dict[str, Any]:
        """Evaluate performance on MCQ tasks."""
        ground_truth = self.evaluation_data.get("ground_truth") 
        
        if not ground_truth:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        for method, result in self.results["decision_results"].items():
            if result and "winning_option" in result:
                selected = result["winning_option"]
                correct = selected == ground_truth
                
                metrics[method] = {
                    "correct": correct,
                    "confidence": result.get("confidence", 0)
                }
        
        return metrics

    def _calculate_ranking_error(self, ranking1: List[str], ranking2: List[str]) -> int:
        """Calculate error between two rankings."""
        pos1 = {item: i for i, item in enumerate(ranking1)}
        pos2 = {item: i for i, item in enumerate(ranking2)}
        
        error = 0
        for item in set(pos1.keys()) & set(pos2.keys()):
            error += abs(pos1[item] - pos2[item])
        
        return error

    def _evaluate_yes_no_maybe_performance(self) -> Dict[str, Any]:
        """Evaluate performance on yes/no/maybe tasks."""
        ground_truth = self.evaluation_data.get("ground_truth").lower()
        
        if not ground_truth or ground_truth not in ["yes", "no", "maybe"]:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        for method, result in self.results["decision_results"].items():
            if result and "winning_option" in result:
                selected = result["winning_option"]
                correct = selected == ground_truth if selected else False
                
                metrics[method] = {
                    "correct": correct,
                    "confidence": result.get("confidence", 0)
                }
        
        return metrics

    def _evaluate_ranking_performance(self) -> Dict[str, Any]:
        """Evaluate performance on ranking tasks."""
        ground_truth = self.evaluation_data.get("ground_truth")
        
        if not ground_truth:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        metrics = {}
        
        for method, result in self.results["decision_results"].items():
            if result and "final_ranking" in result:
                ranking = result["final_ranking"]
                correlation = self._calculate_rank_correlation(ranking, ground_truth)
                error = self._calculate_ranking_error(ranking, ground_truth)
                
                metrics[method] = {
                    "correlation": correlation,
                    "error": error
                }
        
        return metrics

    def _evaluate_teamwork_performance(self) -> Dict[str, Any]:
        """Evaluate performance of the teamwork components."""
        teamwork_metrics = {}
        
        if "closed_loop_communication" in self.results["teamwork_metrics"]:
            comm_metrics = self.results["teamwork_metrics"]["closed_loop_communication"]
            teamwork_metrics["closed_loop_communication"] = {
                "effectiveness": comm_metrics.get("effectiveness_rating", "N/A"),
                "misunderstanding_rate": comm_metrics.get("misunderstanding_rate", 0),
                "total_exchanges": comm_metrics.get("total_exchanges", 0)
            }
        
        if "mutual_monitoring" in self.results["teamwork_metrics"]:
            monitor_metrics = self.results["teamwork_metrics"]["mutual_monitoring"]
            teamwork_metrics["mutual_monitoring"] = {
                "effectiveness": monitor_metrics.get("team_monitoring_effectiveness", "N/A"),
                "issue_resolution_rate": monitor_metrics.get("issue_resolution_rate", 0),
                "total_issues_detected": monitor_metrics.get("total_issues_detected", 0)
            }
        
        if "shared_mental_model" in self.results["teamwork_metrics"]:
            model_metrics = self.results["teamwork_metrics"]["shared_mental_model"]
            teamwork_metrics["shared_mental_model"] = {
                "effectiveness": model_metrics.get("effectiveness_rating", "N/A"),
                "convergence_trend": model_metrics.get("convergence_trend", "unknown"),
                "final_convergence": model_metrics.get("final_convergence", 0)
            }
        
        if self.use_team_leadership:
            leadership_actions = sum(1 for exchange in self.results["exchanges"] 
                                  if exchange.get("type") in ["leadership_definition", "leadership_synthesis"])
            
            teamwork_metrics["leadership"] = {
                "leader_role": self.leader.role if self.leader else "None",
                "leadership_actions": leadership_actions
            }
        
        return teamwork_metrics

    def _calculate_rank_correlation(self, ranking1: List[str], ranking2: List[str]) -> float:
        """Calculate Spearman's rank correlation between two rankings."""
        pos1 = {item: i for i, item in enumerate(ranking1)}
        pos2 = {item: i for i, item in enumerate(ranking2)}
        
        common_items = set(ranking1) & set(ranking2)
        
        if not common_items:
            return 0.0
        
        d_squared_sum = sum((pos1[item] - pos2[item])**2 for item in common_items)
        n = len(common_items)
        
        correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        
        return correlation

    def _evaluate_multi_choice_performance(self) -> Dict[str, Any]:
        """Evaluate performance on multi-choice MCQ tasks."""
        ground_truth = config.TASK.get("ground_truth", "")
        
        if not ground_truth:
            return {"metric": "no_ground_truth", "note": "No ground truth provided for evaluation"}
        
        # Parse ground truth
        if isinstance(ground_truth, str):
            if ',' in ground_truth:
                ground_truth_set = set(opt.strip().upper() for opt in ground_truth.split(','))
            else:
                ground_truth_set = {ground_truth.upper()}
        else:
            ground_truth_set = {str(ground_truth).upper()}
        
        metrics = {}
        
        for method, result in self.results["decision_results"].items():
            if result and "winning_options" in result:
                selected_options = set(result["winning_options"])
                correct = selected_options == ground_truth_set
                
                if ground_truth_set and selected_options:
                    intersection = len(ground_truth_set.intersection(selected_options))
                    union = len(ground_truth_set.union(selected_options))
                    partial_score = intersection / union if union > 0 else 0
                else:
                    partial_score = 0
                
                metrics[method] = {
                    "correct": correct,
                    "partial_score": partial_score,
                    "confidence": result.get("confidence", 0)
                }
            elif result and "winning_option" in result:
                # Handle case where method returned single option for multi-choice
                selected = {result["winning_option"]} if result["winning_option"] else set()
                correct = selected == ground_truth_set
                
                if ground_truth_set and selected:
                    intersection = len(ground_truth_set.intersection(selected))
                    union = len(ground_truth_set.union(selected))
                    partial_score = intersection / union if union > 0 else 0
                else:
                    partial_score = 0
                
                metrics[method] = {
                    "correct": correct,
                    "partial_score": partial_score,
                    "confidence": result.get("confidence", 0)
                }
        
        return metrics