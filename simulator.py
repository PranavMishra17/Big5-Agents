"""
Fixed simulator.py with proper MedRAG integration and knowledge enhancement.
"""

import os
import logging
import json
import re
import traceback
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import copy

from components.modular_agent import MedicalImageAnalyst, ModularAgent, PathologySpecialist, create_agent_team
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
from components.medrag_integration import MedRAGIntegration, create_medrag_integration

class AgentSystemSimulator:
    """
    Enhanced simulator with proper MedRAG integration for parallel question handling.
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
         n_max: int = 5,
         deployment_config: Dict[str, str] = None,
         question_specific_context=False,
         task_config: Dict[str, Any] = None,
         eval_data: Dict[str, Any] = None,
         use_medrag: bool = False):
        """Initialize the simulator with isolated task configuration."""
        
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
        self.question_specific_context = question_specific_context
        
        # Store deployment configuration for this question
        self.deployment_config = deployment_config
        
        # CRITICAL: Store task configuration separately to avoid global state contamination
        self.task_config = task_config or copy.deepcopy(config.TASK)
        self.evaluation_data = eval_data or {}
        
        self.metadata = {}

        # MedRAG Integration - FIXED
        self.use_medrag = use_medrag
        self.medrag_integration = None
        self.retrieved_knowledge = None
        

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
            "task": self.task_config.get("name", "Unknown"),
            "deployment": self.deployment_config['name'] if self.deployment_config else "default",
            "use_medrag": self.use_medrag,
        }
        
        # Setup logging
        self.logger = SimulationLogger(
            simulation_id=self.simulation_id,
            log_dir=config.LOG_DIR,
            config=self.config
        )

        # Initialize MedRAG FIRST (before creating agents)
        if self.use_medrag:
            self._initialize_medrag()

        # Create agent team with isolated context
        self._create_agent_team(isolated_context=question_specific_context)
        
        # Initialize teamwork components
        self.comm_handler = ClosedLoopCommunication() if self.use_closed_loop_comm else None
        self.mutual_monitor = MutualMonitoring() if self.use_mutual_monitoring else None
        self.mental_model = SharedMentalModel() if self.use_shared_mental_model else None
        self.team_orientation = TeamOrientation() if self.use_team_orientation else None
        self.mutual_trust = MutualTrust(self.mutual_trust_factor) if self.use_mutual_trust else None
        
        # Initialize mutual trust network if enabled
        if self.mutual_trust:
            self.mutual_trust.initialize_trust_network(list(self.agents.keys()))
        
        # Initialize decision methods with isolated task config
        self.decision_methods = DecisionMethods(task_config=self.task_config)
        
        # Initialize shared knowledge with isolated task config
        if self.mental_model:
            self.mental_model.initialize_task_model(self.task_config)
            self.mental_model.initialize_team_model(list(self.agents.keys()))
        
        # Store results
        self.results = {
            "simulation_id": self.simulation_id,
            "config": self.config,
            "exchanges": [],
            "decision_results": {}
        }
        
        deployment_name = self.deployment_config['name'] if self.deployment_config else "default"
        self.logger.logger.info(f"Initialized simulation {self.simulation_id} with deployment: {deployment_name}")

    def _initialize_medrag(self):
        """Initialize MedRAG integration."""
        try:
            self.medrag_integration = create_medrag_integration(
                deployment_config=self.deployment_config,
                retriever_name="MedCPT",
                corpus_name="Textbooks"
            )
            
            if not self.medrag_integration:
                self.logger.logger.warning("MedRAG integration not available")
                self.use_medrag = False
                return
            
            # Retrieve knowledge for the current question
            question = self.task_config.get("description", "")
            options = self.task_config.get("options", [])
            
            if question:
                self.logger.logger.info("Retrieving medical knowledge with MedRAG...")
                self.retrieved_knowledge = self.medrag_integration.retrieve_knowledge(
                    question=question,
                    options=options,
                    question_id=self.simulation_id
                )
                
                if self.retrieved_knowledge.get("available", False):
                    num_snippets = len(self.retrieved_knowledge.get("knowledge_snippets", []))
                    self.logger.logger.info(f"MedRAG retrieved {num_snippets} knowledge snippets successfully")
                else:
                    self.logger.logger.warning(f"MedRAG retrieval failed: {self.retrieved_knowledge.get('error', 'Unknown error')}")
            else:
                self.logger.logger.warning("No question available for MedRAG retrieval")
                
        except Exception as e:
            self.logger.logger.error(f"MedRAG initialization failed: {str(e)}")
            self.use_medrag = False
            self.retrieved_knowledge = None

    def _create_agent_team(self, isolated_context: bool = False):
        """Create agent team with proper recruitment handling and deployment assignment."""
        if self.use_recruitment and self.task_config.get("description"):
            try:
                from components.agent_recruitment import determine_complexity, recruit_agents_isolated
                complexity = determine_complexity(self.task_config["description"], self.recruitment_method)
                self.metadata["complexity"] = complexity
                
                # Create teamwork config to pass to recruitment
                teamwork_config = {
                    "use_team_leadership": self.use_team_leadership,
                    "use_closed_loop_comm": self.use_closed_loop_comm,
                    "use_mutual_monitoring": self.use_mutual_monitoring,
                    "use_shared_mental_model": self.use_shared_mental_model,
                    "use_team_orientation": self.use_team_orientation,
                    "use_mutual_trust": self.use_mutual_trust
                }
                
                # Use isolated recruitment that doesn't read from global config
                agents, leader = recruit_agents_isolated(
                    self.task_config["description"],
                    complexity,
                    self.recruitment_pool,
                    self.n_max,
                    self.recruitment_method,
                    self.deployment_config,
                    self.task_config,
                    teamwork_config  # Pass teamwork config
                )
                self.agents = agents
                self.leader = leader
                
            except Exception as e:
                logging.error(f"Recruitment failed: {str(e)}, using default team")
                self.agents, self.leader = self._create_default_team()
        else:
            self.agents, self.leader = self._create_default_team()

    def _create_default_team(self):
        """Create default team with deployment override if specified."""
        if self.deployment_config:
            return self._create_team_with_deployment()
        else:
            from components.modular_agent import create_agent_team_isolated
            return create_agent_team_isolated(
                use_team_leadership=self.use_team_leadership,
                use_closed_loop_comm=self.use_closed_loop_comm,
                use_mutual_monitoring=self.use_mutual_monitoring,
                use_shared_mental_model=self.use_shared_mental_model,
                use_team_orientation=self.use_team_orientation,
                use_mutual_trust=self.use_mutual_trust,
                random_leader=self.random_leader,
                use_recruitment=False,
                n_max=self.n_max,
                deployment_config=self.deployment_config,
                task_config=self.task_config
            )

    def _create_team_with_deployment(self):
        """Create a team where all agents use the specified deployment."""
        from components.modular_agent import ModularAgent
        
        agents = {}
        leader = None
        
        # Create a single agent using the specified deployment
        role = "Medical Generalist"
        agent = ModularAgent(
            role_type=role,
            use_team_leadership=self.use_team_leadership,
            use_closed_loop_comm=self.use_closed_loop_comm,
            use_mutual_monitoring=self.use_mutual_monitoring,
            use_shared_mental_model=self.use_shared_mental_model,
            use_team_orientation=self.use_team_orientation,
            use_mutual_trust=self.use_mutual_trust,
            deployment_config=self.deployment_config,
            agent_index=0,
            task_config=self.task_config  # Pass isolated task config
        )
        
        agents[role] = agent
        if self.use_team_leadership:
            leader = agent
            
        return agents, leader

    def run_simulation(self):
        """
        Run the enhanced 3-round simulation process with MedRAG integration.
        
        Returns:
            Dictionary with simulation results
        """
        # ENHANCEMENT PHASE: Apply MedRAG knowledge if available
        if self.use_medrag and self.retrieved_knowledge:
            self.logger.logger.info("ENHANCEMENT PHASE: Applying retrieved medical knowledge to agents")
            self._enhance_agents_with_retrieved_knowledge()
        else:
            self.results["medrag_enhancement"] = {"enabled": False}

        # ROUND 1: Independent Analysis (Sequential)
        self.logger.logger.info("ROUND 1: Independent task analysis (sequential execution)")
        round1_analyses = self._run_round1_independent_analysis()
        
        # Leadership definition if enabled (between rounds)
        if self.use_team_leadership and self.leader:
            self.logger.logger.info("Leadership phase: Defining task approach")
            self._run_leadership_definition()
        
        # ROUND 2: Collaborative Discussion (Sequential)
        self.logger.logger.info("ROUND 2: Collaborative discussion")
        round2_discussions = self._run_round2_collaborative_discussion(round1_analyses)
        
        # ROUND 3: Final Independent Decision (Sequential)
        self.logger.logger.info("ROUND 3: Final independent decisions (sequential execution)")
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
                "deployment": self.deployment_config['name'] if self.deployment_config else "default",
                "medrag_enhancement": self.results.get("medrag_enhancement", {}),
                "task_info": {
                    "name": self.task_config.get("name", ""),
                    "type": self.task_config.get("type", ""),
                    "description": self.task_config.get("description", "")[:200] + "...",
                    "options": self.task_config.get("options", [])
                }
            },
            "agent_analyses": round1_analyses,      # Round 1 independent analyses
            "agent_responses": round3_decisions,    # Round 3 final decisions
            "exchanges": self.results.get("exchanges", []),
            "decision_results": decision_results
        }


    def _enhance_agents_with_retrieved_knowledge(self):
        """
        Apply retrieved MedRAG knowledge to agents.
        FIXED VERSION - Actually integrates knowledge into agent responses.
        """
        if not self.retrieved_knowledge or not self.retrieved_knowledge.get("available", False):
            self.logger.logger.warning("No valid retrieved knowledge to apply to agents")
            self.results["medrag_enhancement"] = {
                "enabled": True,
                "success": False,
                "error": "No valid knowledge retrieved",
                "agents_enhanced": 0,
                "total_agents": len(self.agents),
                "snippets_retrieved": 0
            }
            return

        enhanced_agents = 0
        
        try:
            # Create comprehensive knowledge context from retrieved snippets
            knowledge_context = self._create_medrag_context()
            
            # Enhance each agent with the knowledge context
            for role, agent in self.agents.items():
                try:
                    # Store the knowledge context in agent's knowledge base
                    agent.add_to_knowledge_base("medrag_knowledge", self.retrieved_knowledge)
                    agent.add_to_knowledge_base("medrag_context", knowledge_context)
                    
                    # CRITICAL FIX: Set a flag that the agent has MedRAG enhancement
                    agent.add_to_knowledge_base("has_medrag_enhancement", True)
                    
                    enhanced_agents += 1
                    self.logger.logger.debug(f"Enhanced {role} with MedRAG knowledge")
                    
                except Exception as e:
                    self.logger.logger.error(f"Failed to enhance {role}: {str(e)}")
            
            # Enhance shared mental model if available
            if self.use_shared_mental_model and self.mental_model:
                success = self.medrag_integration.enhance_shared_mental_model(
                    self.mental_model, 
                    self.retrieved_knowledge
                )
                if success:
                    self.logger.logger.info("Enhanced shared mental model with MedRAG knowledge")
            
            # Log enhancement summary
            num_snippets = len(self.retrieved_knowledge.get("knowledge_snippets", []))
            retrieval_time = self.retrieved_knowledge.get("retrieval_time", 0)
            
            self.logger.logger.info(
                f"MedRAG enhancement completed: {enhanced_agents}/{len(self.agents)} agents enhanced, "
                f"{num_snippets} knowledge snippets applied in {retrieval_time:.2f}s"
            )
            
            # Store for results
            self.results["medrag_enhancement"] = {
                "enabled": True,
                "success": True,
                "agents_enhanced": enhanced_agents,
                "total_agents": len(self.agents),
                "snippets_retrieved": num_snippets,
                "retrieval_time": retrieval_time,
                "summary": self.retrieved_knowledge.get("summary", ""),
                "knowledge_context": knowledge_context[:500] + "..." if len(knowledge_context) > 500 else knowledge_context
            }
            
        except Exception as e:
            error_msg = f"Failed to enhance agents with MedRAG knowledge: {str(e)}"
            self.logger.logger.error(error_msg)
            self.logger.logger.error(traceback.format_exc())
            
            self.results["medrag_enhancement"] = {
                "enabled": True,
                "success": False,
                "error": error_msg,
                "agents_enhanced": enhanced_agents,
                "total_agents": len(self.agents),
                "snippets_retrieved": 0,
                "retrieval_time": 0
            }

    def _create_medrag_context(self) -> str:
        """Create formatted medical knowledge context from retrieved snippets."""
        if not self.retrieved_knowledge or not self.retrieved_knowledge.get("knowledge_snippets"):
            return ""
        
        snippets = self.retrieved_knowledge["knowledge_snippets"]
        context_parts = []
        
        context_parts.append("=== RETRIEVED MEDICAL KNOWLEDGE ===")
        context_parts.append("The following medical literature is relevant to this question:\n")
        
        # Use top 3 most relevant snippets
        top_snippets = sorted(snippets, key=lambda x: x.get("relevance_score", 0), reverse=True)[:3]
        
        for i, snippet in enumerate(top_snippets, 1):
            title = snippet.get("title", f"Medical Reference {i}")
            content = snippet.get("content", "")
            score = snippet.get("relevance_score", 0)
            
            # Truncate very long content
            if len(content) > 300:
                content = content[:300] + "..."
            
            context_parts.append(f"[{i}] {title} (relevance: {score:.2f}):")
            context_parts.append(f"{content}\n")
        
        # Add insights if available
        insights = self.retrieved_knowledge.get("medrag_insights", {})
        if insights.get("reasoning"):
            context_parts.append("Additional clinical reasoning from literature:")
            context_parts.append(f"{insights['reasoning'][:200]}...\n")
        
        context_parts.append("=== END RETRIEVED KNOWLEDGE ===")
        
        return "\n".join(context_parts)

    def _run_round1_independent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        ROUND 1: Each agent analyzes the task independently with enhanced image support.
        """
        agent_analyses = {}
        
        # Extract image from task config if available
        task_image = None
        image_type = "text_only"
        
        if "image_data" in self.task_config:
            image_data = self.task_config["image_data"]
            task_image = image_data.get("image")
            
            # Determine image type for specialized handling
            if image_data.get("is_pathology_image", False):
                image_type = "pathology"
            elif image_data.get("image_type") == "pathology_slide":
                image_type = "pathology"
            elif image_data.get("image_type") == "medical_image":
                image_type = "medical"
            else:
                image_type = "medical"  # Default for any medical image
        
        # Log image analysis mode
        if task_image is not None:
            self.logger.logger.info(f"Round 1: Vision-enabled analysis mode - {image_type} image detected")
        else:
            self.logger.logger.info("Round 1: Text-only analysis mode")
        
        # Process agents sequentially with enhanced error handling
        for role, agent in self.agents.items():
            try:
                self.logger.logger.info(f"Round 1: Getting analysis from {role}")
                
                # Enhanced image analysis with agent type matching
                if task_image is not None:
                    # Check if we have specialized vision agents
                    if isinstance(agent, PathologySpecialist) and image_type == "pathology":
                        self.logger.logger.info(f"{role}: Using specialized pathology analysis")
                        analysis = agent.analyze_pathology_slide(
                            self.task_config["description"], 
                            task_image, 
                            self.task_config
                        )
                    elif isinstance(agent, MedicalImageAnalyst):
                        self.logger.logger.info(f"{role}: Using specialized medical imaging analysis")
                        analysis = agent.analyze_medical_image(
                            self.task_config["description"], 
                            task_image, 
                            self.task_config
                        )
                    else:
                        # General agent with image
                        self.logger.logger.info(f"{role}: Using general vision analysis")
                        analysis = agent.analyze_task_with_image(self.task_config, task_image)
                else:
                    # Text-only analysis
                    analysis = agent.analyze_task_isolated(self.task_config)
                
                # Extract response using isolated task config
                extract = agent.extract_response_isolated(analysis, self.task_config)
                
                # Log to main discussion channel
                self.logger.log_main_discussion(
                    "round1_independent_analysis",
                    role,
                    analysis
                )
                
                # Store analysis with metadata
                agent_analyses[role] = {
                    "analysis": analysis,
                    "extract": extract,
                    "used_vision": task_image is not None,
                    "image_type": image_type,
                    "agent_type": type(agent).__name__
                }
                
                # Update shared mental model if enabled
                if self.use_shared_mental_model and self.mental_model:
                    understanding = self.mental_model.extract_understanding_from_message(analysis)
                    self.mental_model.update_shared_understanding(role, understanding)
                
                # Log MedRAG and vision usage
                medrag_used = agent.get_from_knowledge_base("has_medrag_enhancement")
                if medrag_used and task_image is not None:
                    self.logger.logger.info(f"{role}: Used both MedRAG enhancement and vision analysis")
                elif medrag_used:
                    self.logger.logger.info(f"{role}: Used MedRAG enhancement")
                elif task_image is not None:
                    self.logger.logger.info(f"{role}: Used vision analysis")
                    
            except Exception as e:
                self.logger.logger.error(f"Failed to get analysis from {role}: {str(e)}")
                
                # Enhanced error logging for vision issues
                if task_image is not None and "image" in str(e).lower():
                    self.logger.logger.error(f"Vision-related error for {role}, falling back to text-only")
                    try:
                        # Fallback to text-only analysis
                        fallback_analysis = agent.analyze_task_isolated(self.task_config)
                        fallback_extract = agent.extract_response_isolated(fallback_analysis, self.task_config)
                        
                        agent_analyses[role] = {
                            "analysis": f"[Vision analysis failed, using text-only] {fallback_analysis}",
                            "extract": fallback_extract,
                            "used_vision": False,
                            "vision_error": str(e),
                            "fallback_used": True
                        }
                    except Exception as fallback_e:
                        self.logger.logger.error(f"Fallback also failed for {role}: {str(fallback_e)}")
                        agent_analyses[role] = {
                            "analysis": f"Error occurred: {str(e)}",
                            "extract": {"error": str(e)},
                            "used_vision": False,
                            "vision_error": str(e)
                        }
                else:
                    agent_analyses[role] = {
                        "analysis": f"Error occurred: {str(e)}",
                        "extract": {"error": str(e)},
                        "used_vision": False
                    }
        
        # Log round completion with vision statistics
        vision_count = sum(1 for a in agent_analyses.values() if a.get("used_vision", False))
        error_count = sum(1 for a in agent_analyses.values() if "error" in a.get("extract", {}))
        
        self.logger.logger.info(
            f"Round 1 completed: {len(agent_analyses)} analyses collected, "
            f"{vision_count} used vision, {error_count} errors"
        )
        
        return agent_analyses

    def _run_round2_collaborative_discussion(self, round1_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        ROUND 2: Enhanced collaborative discussion with vision-aware prompting.
        """
        round2_discussions = {}
        
        # Check if this is a vision task
        has_vision_task = any(analysis.get("used_vision", False) for analysis in round1_analyses.values())
        
        if has_vision_task:
            self.logger.logger.info("Round 2: Vision-enabled collaborative discussion")
            # Get fresh image for discussion
            task_image = self._get_fresh_image()
        else:
            self.logger.logger.info("Round 2: Text-only collaborative discussion")
            task_image = None
        
        # Sequential execution to maintain discussion flow
        for role, agent in self.agents.items():
            try:
                self.logger.logger.info(f"Round 2: Collaborative discussion for {role}")
                
                # Collect sanitized analyses from other agents
                other_analyses = {}
                vision_insights = []
                
                for other_role, analysis_data in round1_analyses.items():
                    if other_role != role:
                        # Include analysis but remove final answers
                        analysis_text = analysis_data["analysis"]
                        
                        # Extract vision insights if available
                        if analysis_data.get("used_vision", False):
                            vision_insights.append(f"{other_role} (with image analysis): {analysis_text}")
                        else:
                            other_analyses[other_role] = analysis_text
                
                if len(other_analyses) > 0 or len(vision_insights) > 0:
                    # Create enhanced discussion prompt
                    discussion_parts = [
                        f"""You have completed your initial analysis. Now review your teammates' reasoning to enhance your understanding.

    Your initial analysis:
    {round1_analyses[role]['analysis']}"""
                    ]
                    
                    # Add text-only teammate analyses
                    if other_analyses:
                        other_analyses_text = "\n\n".join([f"{other_role}:\n{analysis}" 
                                                        for other_role, analysis in other_analyses.items()])
                        discussion_parts.append(f"Teammates' reasoning:\n{other_analyses_text}")
                    
                    # Add vision-enhanced insights
                    if vision_insights:
                        vision_text = "\n\n".join(vision_insights)
                        discussion_parts.append(f"Vision-based insights from teammates:\n{vision_text}")
                    
                    # Add discussion instructions
                    discussion_parts.append("""
    Based on these different perspectives:
    1. Identify where you agree or disagree with your teammates
    2. Question any reasoning that seems unclear or potentially flawed
    3. Share additional insights that might help the team""")
                    
                    # Add vision-specific instructions if applicable
                    if has_vision_task:
                        discussion_parts.append("""
    4. If you can see the image, compare your visual observations with teammates' findings
    5. Discuss any visual details that might have been missed or interpreted differently
    6. Consider how image findings support or contradict different reasoning approaches""")
                    
                    discussion_parts.append("""
    DO NOT provide a final answer in this round. Focus on collaborative analysis and discussion.
    This is about improving understanding before making your final decision.""")
                    
                    discussion_prompt = "\n\n".join(discussion_parts)
                    
                    # Apply teamwork components
                    if self.use_mutual_monitoring and self.mutual_monitor:
                        for other_role, analysis in other_analyses.items():
                            other_agent = self.agents[other_role]
                            extract = other_agent.extract_response_isolated(analysis, self.task_config)
                            
                            monitoring_result = self.mutual_monitor.monitor_agent_response(
                                other_role, analysis, None
                            )
                            
                            if monitoring_result["issues_detected"]:
                                feedback = self.mutual_monitor.generate_feedback(
                                    monitoring_result, role
                                )
                                discussion_prompt += f"""

    Based on monitoring {other_role}'s analysis, you've identified:
    {feedback}

    Consider these points in your discussion."""
                    
                    # Execute discussion with or without image
                    if task_image is not None and round1_analyses[role].get("used_vision", False):
                        # Agent used vision in Round 1, so include image in discussion
                        discussion_response = agent.chat_with_image(discussion_prompt, task_image)
                    else:
                        # Text-only discussion
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
                        "message": discussion_response,
                        "used_vision": task_image is not None and round1_analyses[role].get("used_vision", False)
                    })
                    
                    round2_discussions[role] = discussion_response
                    
                else:
                    # Single agent case
                    round2_discussions[role] = "No teammates to discuss with."
                    
            except Exception as e:
                self.logger.logger.error(f"Error in Round 2 discussion for {role}: {str(e)}")
                round2_discussions[role] = f"Error in discussion: {str(e)}"
        
        return round2_discussions

    def _run_round3_final_decisions(self, round1_analyses: Dict[str, Dict[str, Any]], 
                                round2_discussions: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        ROUND 3: Enhanced final decisions with vision support and error recovery.
        """
        round3_decisions = {}
        
        # Check if this is a vision task
        has_vision_task = any(analysis.get("used_vision", False) for analysis in round1_analyses.values())
        
        if has_vision_task:
            self.logger.logger.info("Round 3: Vision-enabled final decisions")
            task_image = self._get_fresh_image()
        else:
            self.logger.logger.info("Round 3: Text-only final decisions")
            task_image = None
        
        # Process agents sequentially
        for role, agent in self.agents.items():
            try:
                task_type = self.task_config.get("type", "mcq")
                
                # Create enhanced final decision prompt
                try:
                    final_prompt = get_adaptive_prompt(
                        "final_decision",
                        task_type,
                        initial_analysis=round1_analyses[role]['analysis'],
                        discussion_summary=round2_discussions.get(role, "No discussion occurred.")
                    )
                except Exception as e:
                    # Enhanced fallback prompts for vision tasks
                    if has_vision_task:
                        if task_type == "yes_no_maybe":
                            final_prompt = f"""Based on your initial image analysis and team discussion, provide your final answer.

    Your initial analysis (with image):
    {round1_analyses[role]['analysis']}

    Team discussion insights:
    {round2_discussions.get(role, "No discussion occurred.")}

    Now examine the image once more and provide your final answer.
    Begin with "ANSWER: X" (yes, no, or maybe) followed by your integrated reasoning."""
                        else:
                            final_prompt = f"""Based on your initial image analysis and team discussion, provide your final answer.

    Your initial analysis (with image):
    {round1_analyses[role]['analysis']}

    Team discussion insights:
    {round2_discussions.get(role, "No discussion occurred.")}

    Now examine the image once more and provide your final answer.
    Begin with "ANSWER: X" (your chosen option) followed by your integrated visual and clinical reasoning."""
                    else:
                        # Standard fallback for text-only
                        final_prompt = f"""Based on your initial analysis and team discussion, provide your final answer.

    Your initial analysis:
    {round1_analyses[role]['analysis']}

    Team discussion insights:
    {round2_discussions.get(role, "No discussion occurred.")}

    Now provide your final answer with "ANSWER: X" followed by your reasoning."""
                
                # Execute final decision with vision support and error recovery
                try:
                    if task_image is not None and round1_analyses[role].get("used_vision", False):
                        # Agent used vision successfully in Round 1
                        final_decision = agent.chat_with_image(final_prompt, task_image)
                    else:
                        # Text-only final decision
                        final_decision = agent.chat(final_prompt)
                except Exception as vision_error:
                    # Vision error recovery
                    if task_image is not None:
                        self.logger.logger.warning(f"Vision error in Round 3 for {role}, falling back to text: {vision_error}")
                        fallback_prompt = f"{final_prompt}\n\n[Note: Image analysis requested but failed. Providing text-based reasoning.]"
                        final_decision = agent.chat(fallback_prompt)
                    else:
                        raise vision_error
                
                # Log to main discussion channel
                self.logger.log_main_discussion(
                    "round3_final_decision",
                    role,
                    final_decision
                )
                
                # Store in results with enhanced metadata
                self.results["exchanges"].append({
                    "type": "round3_final_decision",
                    "communication": "standard",
                    "sender": role,
                    "message": final_decision,
                    "used_vision": task_image is not None and round1_analyses[role].get("used_vision", False)
                })
                
                # Extract the response structure
                extracted = agent.extract_response_isolated(final_decision, self.task_config)
                
                # Get agent weight
                weight = agent.get_from_knowledge_base("weight") or 0.2
                
                # Store enhanced decision data
                round3_decisions[role] = {
                    "final_decision": final_decision,
                    "extract": extracted,
                    "weight": weight,
                    "used_vision_round1": round1_analyses[role].get("used_vision", False),
                    "used_vision_round3": task_image is not None and round1_analyses[role].get("used_vision", False),
                    "agent_type": type(agent).__name__
                }
                
            except Exception as e:
                self.logger.logger.error(f"Failed to get final decision from {role}: {str(e)}")
                round3_decisions[role] = {
                    "final_decision": f"Error occurred: {str(e)}",
                    "extract": {"error": str(e)},
                    "weight": 0.2,
                    "error": str(e)
                }
        
        # Enhanced leadership synthesis with vision support
        if self.use_team_leadership and self.leader:
            try:
                context = "\n\n".join([f"{role}:\n{decision['final_decision']}" 
                                    for role, decision in round3_decisions.items() 
                                    if role != self.leader.role])
                
                # Leader synthesis with vision if available
                if task_image is not None:
                    synthesis_prompt = f"""As team leader, synthesize the team's responses considering both textual reasoning and visual analysis.

    Team responses:
    {context}

    Provide final guidance integrating all perspectives and visual findings."""
                    
                    leader_synthesis = self.leader.chat_with_image(synthesis_prompt, task_image)
                else:
                    leader_synthesis = self.leader.leadership_action_isolated("synthesize", context, self.task_config)
                
                self.logger.log_leadership_action("synthesis", leader_synthesis)
                self.logger.log_main_discussion("leadership_synthesis", self.leader.role, leader_synthesis)
                
                self.results["exchanges"].append({
                    "type": "leadership_synthesis",
                    "communication": "standard",
                    "sender": self.leader.role,
                    "message": leader_synthesis,
                    "used_vision": task_image is not None
                })
                
                # Update leader's decision
                leader_extract = self.leader.extract_response_isolated(leader_synthesis, self.task_config)
                round3_decisions[self.leader.role] = {
                    "final_decision": leader_synthesis,
                    "extract": leader_extract,
                    "weight": round3_decisions[self.leader.role].get("weight", 0.2),
                    "is_leader_synthesis": True
                }
                
            except Exception as e:
                self.logger.logger.error(f"Error in leadership synthesis: {str(e)}")
        
        # Log completion statistics
        vision_count = sum(1 for d in round3_decisions.values() if d.get("used_vision_round3", False))
        error_count = sum(1 for d in round3_decisions.values() if "error" in d)
        
        self.logger.logger.info(
            f"Round 3 completed: {len(round3_decisions)} decisions collected, "
            f"{vision_count} used vision, {error_count} errors"
        )
        
        return round3_decisions

    def _get_fresh_image(self):
        """Get fresh image from task config to avoid corruption between rounds."""
        if "image_data" in self.task_config:
            return self.task_config["image_data"].get("image")
        return None


    def _run_leadership_definition(self) -> str:
        """Have the leader define the team's approach (between Round 1 and 2)."""
        if not self.leader:
            return "No leader designated for this simulation."
        
        try:
            leader_definition = self.leader.leadership_action_isolated("define_task", task_config=self.task_config)
            
            self.logger.log_main_discussion("leadership_definition", self.leader.role, leader_definition)
            self.logger.log_leadership_action("task_definition", leader_definition)
            
            self.results["exchanges"].append({
                "type": "leadership_definition",
                "communication": "standard",
                "sender": self.leader.role,
                "message": leader_definition
            })
            
            return leader_definition
            
        except Exception as e:
            self.logger.logger.error(f"Error in leadership definition: {str(e)}")
            return f"Error in leadership definition: {str(e)}"

    def _apply_decision_methods(self, agent_decisions):
        """Apply decision methods to agent decisions using isolated task config."""
        # Determine if this is an MDT (advanced) task
        is_mdt_task = any(any(prefix in agent_role for prefix in ["1_", "2_", "3_"]) 
                        for agent_role in agent_decisions.keys())
        
        if is_mdt_task:
            logging.info("Using MDT decision process for advanced query")
            return self._apply_mdt_decision_methods(agent_decisions)
        else:
            return self._apply_standard_decision_methods(agent_decisions)

    def _apply_standard_decision_methods(self, agent_decisions):
        """Apply standard decision methods using isolated task config."""
        logging.info(f"Applying standard decision methods to {len(agent_decisions)} agent responses")
        
        # Pass the isolated task config to decision methods
        majority_result = self.decision_methods.majority_voting(agent_decisions, task_config=self.task_config)
        weighted_result = self.decision_methods.weighted_voting(agent_decisions, task_config=self.task_config)
        borda_result = self.decision_methods.borda_count(agent_decisions, task_config=self.task_config)
        
        return {
            "majority_voting": majority_result,
            "weighted_voting": weighted_result,
            "borda_count": borda_result
        }

    def _apply_mdt_decision_methods(self, agent_decisions):
        """Apply decision methods for MDT (advanced) tasks using isolated task config."""
        # Group agents by team
        team_decisions = {}
        for agent_role, response in agent_decisions.items():
            parts = agent_role.split("_")
            if len(parts) >= 2:
                team_name = "_".join(parts[:2])
                if team_name not in team_decisions:
                    team_decisions[team_name] = {}
                team_decisions[team_name][agent_role] = response
        
        # Make team-level decisions using isolated task config
        team_results = {}
        for team_name, team_responses in team_decisions.items():
            team_results[team_name] = self.decision_methods.majority_voting(team_responses, task_config=self.task_config)
        
        # Use final team's decisions
        final_team_name = None
        for team_name in sorted(team_decisions.keys(), reverse=True):
            if team_name.startswith("3_") or "Final" in team_name:
                final_team_name = team_name
                break
        
        final_agent_responses = team_decisions.get(final_team_name, agent_decisions)
        
        # Apply decision methods to final team using isolated task config
        majority_result = self.decision_methods.majority_voting(final_agent_responses, task_config=self.task_config)
        weighted_result = self.decision_methods.weighted_voting(final_agent_responses, task_config=self.task_config)
        borda_result = self.decision_methods.borda_count(final_agent_responses, task_config=self.task_config)
        
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
        task_type = self.task_config["type"]
        
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
        ground_truth = self.evaluation_data.get("ground_truth")
        if ground_truth:
            ground_truth = ground_truth.lower()
        
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
        ground_truth = self.evaluation_data.get("ground_truth", "")
        
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
    
    ##################### =================================================== Image Vision Enhancements =================================================== #####################


    # UPDATE: Add vision analysis method to ModularAgent
    def analyze_task_with_image(self, task_config: Dict[str, Any], image) -> str:
        """
        Analyze task with image using vision capabilities.
        
        Args:
            task_config: Task configuration with image data
            image: PIL Image object
            
        Returns:
            Analysis incorporating visual findings
        """
        task_type = task_config["type"]
        question = task_config["description"]
        
        # Check if this agent is vision-specialized
        if isinstance(self, (MedicalImageAnalyst, PathologySpecialist)):
            if isinstance(self, MedicalImageAnalyst):
                return self.analyze_medical_image(question, image, task_config)
            elif isinstance(self, PathologySpecialist):
                return self.analyze_pathology_slide(question, image, task_config)
        
        # For general agents, use enhanced vision prompt
        vision_prompt = f"""
    You are analyzing a medical question that includes an image. Please examine the provided image carefully and incorporate your visual findings into your analysis.

    Question: {question}

    Please provide:
    1. **Visual Analysis**: Describe what you observe in the image
    2. **Medical Assessment**: Interpret the medical significance of your visual findings  
    3. **Clinical Reasoning**: Connect the image findings to the question asked
    4. **Answer Determination**: Use both the question text and image analysis to determine your answer

    Provide your final answer clearly at the end.
    """
        
        return self.chat_with_image(vision_prompt, image)