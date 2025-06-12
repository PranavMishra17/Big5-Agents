"""
Enhanced logging functionality for agent system with question-level parallel processing support.
"""

import os
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

class SimulationLogger:
    """Enhanced logger for tracking simulation progress and results with deployment info."""
    
    def __init__(self, 
                simulation_id: str, 
                log_dir: str,
                config: Dict[str, bool] = None):
        """
        Initialize the simulation logger.
        
        Args:
            simulation_id: ID for the simulation
            log_dir: Directory to store logs
            config: Configuration options (leadership, closed_loop, etc.)
        """
        self.simulation_id = simulation_id
        self.log_dir = log_dir
        self.config = config or {}
        
        # Create configuration string for the folder name
        config_str = []
        if self.config.get("use_team_leadership"):
            config_str.append("leadership")
        if self.config.get("use_closed_loop_comm"):
            config_str.append("closed_loop")
        if self.config.get("use_mutual_monitoring"):
            config_str.append("mutual_monitoring")
        if self.config.get("use_shared_mental_model"):
            config_str.append("shared_mental_model")
        if self.config.get("use_team_orientation"):
            config_str.append("team_orientation")
        if self.config.get("use_mutual_trust"):
            config_str.append("mutual_trust")
        if self.config.get("use_recruitment"):
            config_str.append("recruitment")
        self.config_name = "_".join(config_str) if config_str else "baseline"
        
        # Create folder structure: logs/[config_name]/[simulation_id]/
        self.run_dir = os.path.join(self.log_dir, self.config_name, self.simulation_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup file paths
        self.log_file = os.path.join(self.run_dir, f"{simulation_id}.log")
        self.events_file = os.path.join(self.run_dir, f"{simulation_id}_events.jsonl")
        
        # Initialize the logger
        self.logger = self._setup_logger()
        
        # Log initial configuration including deployment info
        initial_config = self.config.copy()
        deployment_name = self.config.get("deployment", "default")
        initial_config["deployment_used"] = deployment_name
        
        self.log_event("simulation_started", {
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat(),
            "config": initial_config,
            "config_name": self.config_name,
            "deployment": deployment_name,
            "processing_mode": "question_level_parallel" if len(os.environ.get('AZURE_DEPLOYMENTS', '').split(',')) > 1 else "sequential"
        })
        
        self.logger.info(f"SimulationLogger initialized for {simulation_id} with configuration: {self.config_name}, deployment: {deployment_name}")

        # Create additional log files for different channels
        self.main_discussion_file = os.path.join(self.run_dir, f"{simulation_id}_main_discussion.jsonl")
        self.closed_loop_file = os.path.join(self.run_dir, f"{simulation_id}_closed_loop.jsonl")
        self.leadership_file = os.path.join(self.run_dir, f"{simulation_id}_leadership.jsonl")
        self.monitoring_file = os.path.join(self.run_dir, f"{simulation_id}_monitoring.jsonl")
        self.mental_model_file = os.path.join(self.run_dir, f"{simulation_id}_mental_model.jsonl")
        self.decision_file = os.path.join(self.run_dir, f"{simulation_id}_decision.jsonl")
        self.team_orientation_file = os.path.join(self.run_dir, f"{simulation_id}_team_orientation.jsonl")
        self.mutual_trust_file = os.path.join(self.run_dir, f"{simulation_id}_mutual_trust.jsonl")

    
    def _setup_logger(self) -> logging.Logger:
        """Set up the file and console loggers."""
        logger = logging.getLogger(f"simulation.{self.simulation_id}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler (reduced verbosity for parallel processing)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
        ))
        # Set higher log level for console to reduce noise during parallel processing
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        
        return logger
    

    def log_agent_message(self, agent_role: str, message_type: str, content: str) -> None:
        """
        Log a complete agent message with deployment info.
        
        Args:
            agent_role: Role of the agent (leader/member)
            message_type: Type of message (send, receive, etc.)
            content: Complete message text
        """
        # Log to standard logger with brief message
        deployment = self.config.get("deployment", "default")
        self.logger.info(f"Agent {agent_role} {message_type} (length: {len(content)}) [deployment: {deployment}]")
        
        # Also log as structured event
        self.log_event("agent_message", {
            "agent_role": agent_role,
            "message_type": message_type,
            "content_length": len(content),
            "deployment": deployment
        })
    

    def log_simulation_complete(self, results: Dict[str, Any]) -> None:
        """
        Log the completion of the simulation.
        
        Args:
            results: Final simulation results
        """
        # Create a summary of the results (without the full exchanges)
        results_summary = {k: v for k, v in results.items() if k != "exchanges"}
        deployment = self.config.get("deployment", "default")
        
        self.log_event("simulation_completed", {
            "simulation_id": self.simulation_id,
            "summary": results_summary,
            "deployment": deployment
        })
        
        self.logger.info(f"Simulation {self.simulation_id} completed successfully [deployment: {deployment}]")


    def log_event(self, event_type, data):
        """Log a structured event to the events file with deployment info."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "data": data
        }
        
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Reduced verbosity for parallel processing
        if event_type in ["simulation_started", "simulation_completed"]:
            self.logger.info(f"Event logged: {event_type}")
        else:
            self.logger.debug(f"Event logged: {event_type}")
    

    def log_main_discussion(self, stage, agent_role, message):
        """
        Log main discussion between agents with deployment info.
        
        Args:
            stage: Current discussion stage (task_analysis, collaborative_discussion, etc.)
            agent_role: Role of the agent speaking
            message: Full message text
        """
        event = {
            "stage": stage,
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "message": message
        }
        
        with open(self.main_discussion_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Main discussion: {stage} - {agent_role} [deployment: {self.config.get('deployment', 'default')}]")
    

    def log_closed_loop(self, stage, sender_role, receiver_role, initial_message, acknowledgment, verification):
        """
        Log closed-loop communication events with deployment info.
        
        Args:
            stage: Current discussion stage
            sender_role: Role of the sending agent
            receiver_role: Role of the receiving agent
            initial_message: Initial message sent
            acknowledgment: Receiver's acknowledgment
            verification: Sender's verification
        """
        event = {
            "stage": stage,
            "sender_role": sender_role,
            "receiver_role": receiver_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "initial_message": initial_message,
            "acknowledgment": acknowledgment,
            "verification": verification
        }
        
        with open(self.closed_loop_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Closed-loop communication: {stage} - {sender_role} -> {receiver_role} [deployment: {self.config.get('deployment', 'default')}]")
    

    def log_leadership_action(self, action_type, content):
        """
        Log leader-specific actions with deployment info.
        
        Args:
            action_type: Type of leadership action (task_definition, synthesis, etc.)
            content: Full message content
        """
        event = {
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "content": content
        }
        
        with open(self.leadership_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Leadership action: {action_type} [deployment: {self.config.get('deployment', 'default')}]")
        

    def log_monitoring_action(self, monitor_role, target_role, issues, feedback):
        """
        Log mutual monitoring actions and feedback with deployment info.
        
        Args:
            monitor_role: Role of the monitoring agent
            target_role: Role of the monitored agent
            issues: List of issues detected
            feedback: Feedback provided
        """
        event = {
            "monitor_role": monitor_role,
            "target_role": target_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "issues_detected": len(issues) > 0,
            "issues": issues,
            "feedback": feedback
        }
        
        with open(self.monitoring_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Monitoring action: {monitor_role} monitored {target_role}, issues: {len(issues)} [deployment: {self.config.get('deployment', 'default')}]")
        

    def log_mental_model_update(self, agent_role, understanding, convergence):
        """
        Log shared mental model updates and convergence metrics with deployment info.
        
        Args:
            agent_role: Role of the agent
            understanding: The agent's understanding
            convergence: Current convergence metric
        """
        event = {
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "understanding": understanding,
            "convergence": convergence
        }
        
        with open(self.mental_model_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Mental model update from {agent_role}, convergence: {convergence:.2f} [deployment: {self.config.get('deployment', 'default')}]")
    
    
    def log_decision_output(self, method, result):
        """
        Log decision method outputs with deployment info.
        
        Args:
            method: Decision method used (majority_voting, weighted_voting, borda_count)
            result: Result of the decision method
        """
        event = {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "result": result
        }
        
        with open(self.decision_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # For ranking tasks, log the top item
        if "final_ranking" in result:
            top_item = result["final_ranking"][0] if result["final_ranking"] else "None"
            self.logger.info(f"Decision output: {method} - Top item: {top_item} [deployment: {self.config.get('deployment', 'default')}]")
        # For MCQ tasks, log the selected option
        elif "winning_option" in result:
            self.logger.info(f"Decision output: {method} - Selected: {result['winning_option']} [deployment: {self.config.get('deployment', 'default')}]")
        # For other tasks
        else:
            self.logger.info(f"Decision output: {method} - Result logged [deployment: {self.config.get('deployment', 'default')}]")


    def log_team_orientation_action(self, agent_role, action_type, details):
        """
        Log team orientation actions with deployment info.
        
        Args:
            agent_role: Role of the agent
            action_type: Type of team orientation action
            details: Details of the action
        """
        event = {
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "action_type": action_type,
            "details": details
        }
        
        with open(self.team_orientation_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Team orientation action: {action_type} by {agent_role} [deployment: {self.config.get('deployment', 'default')}]")

    def log_mutual_trust_event(self, from_role, to_role, event_type, trust_level, details):
        """
        Log mutual trust events with deployment info.
        
        Args:
            from_role: Role of the agent giving trust
            to_role: Role of the agent receiving trust
            event_type: Type of trust event
            trust_level: Current trust level
            details: Details of the event
        """
        event = {
            "from_role": from_role,
            "to_role": to_role,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "event_type": event_type,
            "trust_level": trust_level,
            "details": details
        }
        
        with open(self.mutual_trust_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.debug(f"Mutual trust event: {from_role} -> {to_role}, {event_type}, trust: {trust_level:.2f} [deployment: {self.config.get('deployment', 'default')}]")

    def log_parallel_processing_info(self, question_index: int, total_questions: int, batch_info: Dict[str, Any] = None):
        """
        Log information about parallel processing progress.
        
        Args:
            question_index: Current question index
            total_questions: Total number of questions
            batch_info: Information about the current batch
        """
        event = {
            "event_type": "parallel_processing_progress",
            "question_index": question_index,
            "total_questions": total_questions,
            "progress_percentage": (question_index + 1) / total_questions * 100,
            "deployment": self.config.get("deployment", "default"),
            "timestamp": datetime.now().isoformat()
        }
        
        if batch_info:
            event["batch_info"] = batch_info
        
        self.log_event("parallel_processing_progress", event)
        
        # Log progress at INFO level for visibility
        if (question_index + 1) % 10 == 0 or question_index == 0:  # Log every 10 questions or first question
            self.logger.info(f"Progress: {question_index + 1}/{total_questions} questions ({event['progress_percentage']:.1f}%) [deployment: {self.config.get('deployment', 'default')}]")