"""
Enhanced logging functionality for agent system.
"""

import os
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

class SimulationLogger:
    """Enhanced logger for tracking simulation progress and results."""
    
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
        
        # Log initial configuration
        self.log_event("simulation_started", {
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "config_name": self.config_name
        })
        
        self.logger.info(f"SimulationLogger initialized for {simulation_id} with configuration: {self.config_name}")

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
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
        
        return logger
    

    def log_agent_message(self, agent_role: str, message_type: str, content: str) -> None:
        """
        Log a complete agent message.
        
        Args:
            agent_role: Role of the agent (leader/member)
            message_type: Type of message (send, receive, etc.)
            content: Complete message text
        """
        # Log to standard logger with brief message
        self.logger.info(f"Agent {agent_role} {message_type} (length: {len(content)})")
        
        # Also log as structured event
        self.log_event("agent_message", {
            "agent_role": agent_role,
            "message_type": message_type,
            "content_length": len(content)
        })
    

    def log_simulation_complete(self, results: Dict[str, Any]) -> None:
        """
        Log the completion of the simulation.
        
        Args:
            results: Final simulation results
        """
        # Create a summary of the results (without the full exchanges)
        results_summary = {k: v for k, v in results.items() if k != "exchanges"}
        
        self.log_event("simulation_completed", {
            "simulation_id": self.simulation_id,
            "summary": results_summary
        })
        
        self.logger.info(f"Simulation {self.simulation_id} completed successfully")


    def log_event(self, event_type, data):
        """Log a structured event to the events file."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Event logged: {event_type}")
    

    def log_main_discussion(self, stage, agent_role, message):
        """
        Log main discussion between agents.
        
        Args:
            stage: Current discussion stage (task_analysis, collaborative_discussion, etc.)
            agent_role: Role of the agent speaking
            message: Full message text
        """
        event = {
            "stage": stage,
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        with open(self.main_discussion_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Main discussion: {stage} - {agent_role}")
    

    def log_closed_loop(self, stage, sender_role, receiver_role, initial_message, acknowledgment, verification):
        """
        Log closed-loop communication events.
        
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
            "initial_message": initial_message,
            "acknowledgment": acknowledgment,
            "verification": verification
        }
        
        with open(self.closed_loop_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Closed-loop communication: {stage} - {sender_role} -> {receiver_role}")
    

    def log_leadership_action(self, action_type, content):
        """
        Log leader-specific actions.
        
        Args:
            action_type: Type of leadership action (task_definition, synthesis, etc.)
            content: Full message content
        """
        event = {
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "content": content
        }
        
        with open(self.leadership_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Leadership action: {action_type}")
        

    def log_monitoring_action(self, monitor_role, target_role, issues, feedback):
        """
        Log mutual monitoring actions and feedback.
        
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
            "issues_detected": len(issues) > 0,
            "issues": issues,
            "feedback": feedback
        }
        
        with open(self.monitoring_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Monitoring action: {monitor_role} monitored {target_role}, issues: {len(issues)}")
        

    def log_mental_model_update(self, agent_role, understanding, convergence):
        """
        Log shared mental model updates and convergence metrics.
        
        Args:
            agent_role: Role of the agent
            understanding: The agent's understanding
            convergence: Current convergence metric
        """
        event = {
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "understanding": understanding,
            "convergence": convergence
        }
        
        with open(self.mental_model_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Mental model update from {agent_role}, convergence: {convergence:.2f}")
    
    
    def log_decision_output(self, method, result):
        """
        Log decision method outputs.
        
        Args:
            method: Decision method used (majority_voting, weighted_voting, borda_count)
            result: Result of the decision method
        """
        event = {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        with open(self.decision_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # For ranking tasks, log the top item
        if "final_ranking" in result:
            top_item = result["final_ranking"][0] if result["final_ranking"] else "None"
            self.logger.info(f"Decision output: {method} - Top item: {top_item}")
        # For MCQ tasks, log the selected option
        elif "winning_option" in result:
            self.logger.info(f"Decision output: {method} - Selected: {result['winning_option']}")
        # For other tasks
        else:
            self.logger.info(f"Decision output: {method} - Result logged")


    def log_team_orientation_action(self, agent_role, action_type, details):
        """
        Log team orientation actions.
        
        Args:
            agent_role: Role of the agent
            action_type: Type of team orientation action
            details: Details of the action
        """
        event = {
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details
        }
        
        with open(self.team_orientation_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Team orientation action: {action_type} by {agent_role}")

    def log_mutual_trust_event(self, from_role, to_role, event_type, trust_level, details):
        """
        Log mutual trust events.
        
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
            "event_type": event_type,
            "trust_level": trust_level,
            "details": details
        }
        
        with open(self.mutual_trust_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Mutual trust event: {from_role} -> {to_role}, {event_type}, trust: {trust_level:.2f}")