"""
Enhanced logging functionality for agent system with question-level parallel processing support.
Each question gets its own isolated log files.
"""

import os
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

class SimulationLogger:
    """Enhanced logger for tracking simulation progress and results with deployment info and question isolation."""
    
    def __init__(self, 
                simulation_id: str, 
                log_dir: str,
                config: Dict[str, bool] = None,
                question_index: Optional[int] = None):
        """
        Initialize the simulation logger with question-specific isolation.
        
        Args:
            simulation_id: ID for the simulation
            log_dir: Directory to store logs
            config: Configuration options (leadership, closed_loop, etc.)
            question_index: Optional question index for parallel processing
        """
        self.simulation_id = simulation_id
        self.log_dir = log_dir
        self.config = config or {}
        self.question_index = question_index
        
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
        
        # Create folder structure for question-level isolation
        if question_index is not None:
            # For parallel processing: logs/[config_name]/question_[index]/[simulation_id]/
            self.run_dir = os.path.join(self.log_dir, self.config_name, f"question_{question_index}", self.simulation_id)
        else:
            # For single questions: logs/[config_name]/[simulation_id]/
            self.run_dir = os.path.join(self.log_dir, self.config_name, self.simulation_id)
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup file paths
        self.log_file = os.path.join(self.run_dir, f"{simulation_id}.log")
        self.events_file = os.path.join(self.run_dir, f"{simulation_id}_events.jsonl")
        
        # Initialize the logger with question-specific isolation
        self.logger = self._setup_logger()
        
        # Log initial configuration including deployment info
        initial_config = self.config.copy()
        deployment_name = self.config.get("deployment", "default")
        initial_config["deployment_used"] = deployment_name
        
        log_event_data = {
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat(),
            "config": initial_config,
            "config_name": self.config_name,
            "deployment": deployment_name,
            "processing_mode": "question_level_parallel" if question_index is not None else "sequential"
        }
        
        if question_index is not None:
            log_event_data["question_index"] = question_index
        
        self.log_event("simulation_started", log_event_data)
        
        log_msg = f"SimulationLogger initialized for {simulation_id} with configuration: {self.config_name}, deployment: {deployment_name}"
        if question_index is not None:
            log_msg += f", question: {question_index}"
        self.logger.info(log_msg)

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
        """Set up the file and console loggers with question-specific isolation."""
        # Create unique logger name to avoid conflicts between parallel questions
        logger_name = f"simulation.{self.simulation_id}"
        if self.question_index is not None:
            logger_name += f".q{self.question_index}"
            
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any to avoid duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler - each question gets its own log file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler (reduced verbosity for parallel processing)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [Q%(question_index)s-%(name)s] %(message)s' 
            if self.question_index is not None else 
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        # Set higher log level for console to reduce noise during parallel processing
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
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
        log_msg = f"Agent {agent_role} {message_type} (length: {len(content)}) [deployment: {deployment}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.info(log_msg)
        
        # Also log as structured event
        event_data = {
            "agent_role": agent_role,
            "message_type": message_type,
            "content_length": len(content),
            "deployment": deployment
        }
        
        if self.question_index is not None:
            event_data["question_index"] = self.question_index
            
        self.log_event("agent_message", event_data)
    

    def log_simulation_complete(self, results: Dict[str, Any]) -> None:
        """
        Log the completion of the simulation.
        
        Args:
            results: Final simulation results
        """
        # Create a summary of the results (without the full exchanges)
        results_summary = {k: v for k, v in results.items() if k != "exchanges"}
        deployment = self.config.get("deployment", "default")
        
        event_data = {
            "simulation_id": self.simulation_id,
            "summary": results_summary,
            "deployment": deployment
        }
        
        if self.question_index is not None:
            event_data["question_index"] = self.question_index
        
        self.log_event("simulation_completed", event_data)
        
        log_msg = f"Simulation {self.simulation_id} completed successfully [deployment: {deployment}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.info(log_msg)


    def log_event(self, event_type, data):
        """Log a structured event to the events file with deployment info."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "deployment": self.config.get("deployment", "default"),
            "data": data
        }
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Reduced verbosity for parallel processing
        if event_type in ["simulation_started", "simulation_completed"]:
            log_msg = f"Event logged: {event_type}"
            if self.question_index is not None:
                log_msg = f"Q{self.question_index}: {log_msg}"
            self.logger.info(log_msg)
        else:
            log_msg = f"Event logged: {event_type}"
            if self.question_index is not None:
                log_msg = f"Q{self.question_index}: {log_msg}"
            self.logger.debug(log_msg)
    

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.main_discussion_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Main discussion: {stage} - {agent_role} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)
    

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.closed_loop_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Closed-loop communication: {stage} - {sender_role} -> {receiver_role} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)
    

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.leadership_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Leadership action: {action_type} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)
        

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.monitoring_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Monitoring action: {monitor_role} monitored {target_role}, issues: {len(issues)} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)
        

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.mental_model_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Mental model update from {agent_role}, convergence: {convergence:.2f} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)
    
    
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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.decision_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # For ranking tasks, log the top item
        if "final_ranking" in result:
            top_item = result["final_ranking"][0] if result["final_ranking"] else "None"
            log_msg = f"Decision output: {method} - Top item: {top_item} [deployment: {self.config.get('deployment', 'default')}]"
        # For MCQ tasks, log the selected option
        elif "winning_option" in result:
            log_msg = f"Decision output: {method} - Selected: {result['winning_option']} [deployment: {self.config.get('deployment', 'default')}]"
        # For other tasks
        else:
            log_msg = f"Decision output: {method} - Result logged [deployment: {self.config.get('deployment', 'default')}]"
        
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.info(log_msg)


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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.team_orientation_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Team orientation action: {action_type} by {agent_role} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)

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
        
        if self.question_index is not None:
            event["question_index"] = self.question_index
        
        with open(self.mutual_trust_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        log_msg = f"Mutual trust event: {from_role} -> {to_role}, {event_type}, trust: {trust_level:.2f} [deployment: {self.config.get('deployment', 'default')}]"
        if self.question_index is not None:
            log_msg = f"Q{self.question_index}: {log_msg}"
        self.logger.debug(log_msg)

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
            log_msg = f"Progress: {question_index + 1}/{total_questions} questions ({event['progress_percentage']:.1f}%) [deployment: {self.config.get('deployment', 'default')}]"
            if self.question_index is not None:
                log_msg = f"Q{self.question_index}: {log_msg}"
            self.logger.info(log_msg)

    def save_question_result(self, question_result: Dict[str, Any], output_dir: str):
        """
        Save individual question result with proper isolation.
        
        Args:
            question_result: Result data for this question
            output_dir: Output directory for results
        """
        if self.question_index is not None:
            # Create question-specific result file
            result_filename = f"question_{self.question_index}_result.json"
        else:
            result_filename = f"{self.simulation_id}_result.json"
        
        result_path = os.path.join(output_dir, result_filename)
        
        # Add logging metadata to the result
        question_result["logging_info"] = {
            "log_directory": self.run_dir,
            "simulation_id": self.simulation_id,
            "question_index": self.question_index,
            "config_name": self.config_name,
            "deployment": self.config.get("deployment", "default")
        }
        
        try:
            with open(result_path, 'w') as f:
                json.dump(question_result, f, indent=2)
            
            log_msg = f"Saved question result to {result_path}"
            if self.question_index is not None:
                log_msg = f"Q{self.question_index}: {log_msg}"
            self.logger.info(log_msg)
            
        except Exception as e:
            log_msg = f"Failed to save question result: {str(e)}"
            if self.question_index is not None:
                log_msg = f"Q{self.question_index}: {log_msg}"
            self.logger.error(log_msg)

    def get_log_directory(self) -> str:
        """Get the log directory path for this simulation."""
        return self.run_dir

    def cleanup(self):
        """Clean up resources and close log handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)