"""
Token counting utility for tracking LLM API usage with tiktoken.
Provides comprehensive token tracking for input, output, and total usage.
"""
import tiktoken
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging


@dataclass
class TokenUsage:
    """Data class for tracking token usage."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    agent_role: Optional[str] = None
    question_id: Optional[str] = None
    simulation_id: Optional[str] = None
    operation_type: Optional[str] = None  # e.g., 'analysis', 'decision', 'chat', 'medrag'


class TokenCounter:
    """
    Comprehensive token counter with storage and logging capabilities.
    Thread-safe for parallel processing.
    """
    
    def __init__(self, output_dir: str = "token_logs", max_input_tokens: int = 100000, max_output_tokens: int = 8192):
        """
        Initialize token counter.
        
        Args:
            output_dir: Directory to store token logs
            max_input_tokens: Maximum allowed input tokens per call
            max_output_tokens: Maximum allowed output tokens per call
        """
        self.output_dir = output_dir
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self._session_usage: List[TokenUsage] = []
        self._total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize encoders for different models
        self._encoders = {}
        self._initialize_encoders()
        
    def _initialize_encoders(self):
        """Initialize tiktoken encoders for common models."""
        try:
            # GPT-4 and GPT-4o use cl100k_base encoding
            self._encoders["gpt-4"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-4o"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-4-turbo"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-3.5-turbo"] = tiktoken.get_encoding("cl100k_base")
            
            self.logger.info("Initialized tiktoken encoders for token counting")
        except Exception as e:
            self.logger.error(f"Failed to initialize tiktoken encoders: {e}")
            
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get the appropriate encoder for a model."""
        # Normalize model name
        model_key = model.lower()
        if "gpt-4o" in model_key:
            return self._encoders.get("gpt-4o")
        elif "gpt-4" in model_key:
            return self._encoders.get("gpt-4")
        elif "gpt-3.5" in model_key:
            return self._encoders.get("gpt-3.5-turbo")
        else:
            # Default to gpt-4 encoding
            return self._encoders.get("gpt-4", tiktoken.get_encoding("cl100k_base"))
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text using appropriate encoder."""
        try:
            encoder = self.get_encoder(model)
            if encoder:
                return len(encoder.encode(text))
            else:
                # Fallback estimation: ~4 characters per token
                return len(text) // 4
        except Exception as e:
            self.logger.warning(f"Token counting failed, using estimation: {e}")
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, Any]], model: str = "gpt-4o") -> int:
        """
        Count tokens in a list of messages, accounting for chat format overhead.
        Based on OpenAI's token counting guidelines.
        """
        try:
            encoder = self.get_encoder(model)
            num_tokens = 0
            
            for message in messages:
                # Account for message formatting tokens
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    if isinstance(value, str):
                        num_tokens += len(encoder.encode(value))
                    elif isinstance(value, list):  # For vision messages
                        for item in value:
                            if isinstance(item, dict) and "text" in item:
                                num_tokens += len(encoder.encode(item["text"]))
                            elif isinstance(item, dict) and item.get("type") == "image_url":
                                # Estimate tokens for image (vision models)
                                num_tokens += 85  # Base tokens for image
                        
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
            
        except Exception as e:
            self.logger.warning(f"Message token counting failed, using estimation: {e}")
            # Fallback: estimate based on total text length
            total_text = ""
            for message in messages:
                if isinstance(message.get("content"), str):
                    total_text += message["content"]
                elif isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if isinstance(item, dict) and "text" in item:
                            total_text += item["text"]
            return len(total_text) // 4 + len(messages) * 10  # Add overhead per message
    
    def validate_token_limits(self, input_tokens: int, model: str = "gpt-4o") -> Tuple[bool, str]:
        """
        Validate if token count is within limits.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if input_tokens > self.max_input_tokens:
            return False, f"Input tokens ({input_tokens}) exceed limit ({self.max_input_tokens})"
        
        # Model-specific context limits
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385
        }
        
        model_key = model.lower()
        for key, limit in model_limits.items():
            if key in model_key:
                if input_tokens > limit * 0.8:  # Use 80% of context limit for safety
                    return False, f"Input tokens ({input_tokens}) too close to model limit ({limit})"
                break
        
        return True, "Token count within limits"
    
    def track_api_call(self, input_tokens: int, output_tokens: int, model: str, 
                      agent_role: str = None, question_id: str = None, 
                      simulation_id: str = None, operation_type: str = None) -> TokenUsage:
        """
        Track an API call's token usage.
        
        Returns:
            TokenUsage object with the recorded usage
        """
        total_tokens = input_tokens + output_tokens
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            timestamp=datetime.now().isoformat(),
            agent_role=agent_role,
            question_id=question_id,
            simulation_id=simulation_id,
            operation_type=operation_type
        )
        
        with self._lock:
            self._session_usage.append(usage)
            self._total_usage["input_tokens"] += input_tokens
            self._total_usage["output_tokens"] += output_tokens
            self._total_usage["total_tokens"] += total_tokens
            self._total_usage["api_calls"] += 1
        
        self.logger.debug(f"Tracked API call: {input_tokens} in, {output_tokens} out, {total_tokens} total")
        return usage
    
    def get_session_usage(self) -> Dict[str, Any]:
        """Get current session usage statistics."""
        with self._lock:
            return {
                "total_usage": self._total_usage.copy(),
                "call_count": len(self._session_usage),
                "detailed_calls": [asdict(usage) for usage in self._session_usage]
            }
    
    def get_question_usage(self, question_id: str) -> Dict[str, Any]:
        """Get token usage for a specific question."""
        with self._lock:
            question_calls = [usage for usage in self._session_usage if usage.question_id == question_id]
            
            if not question_calls:
                return {"error": f"No usage found for question {question_id}"}
            
            total_input = sum(call.input_tokens for call in question_calls)
            total_output = sum(call.output_tokens for call in question_calls)
            
            return {
                "question_id": question_id,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "api_calls": len(question_calls),
                "detailed_calls": [asdict(usage) for usage in question_calls]
            }
    
    def save_session_usage(self, run_id: str = None) -> str:
        """
        Save session usage to JSON file.
        
        Returns:
            Path to saved file
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"token_usage_{run_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        usage_data = self.get_session_usage()
        usage_data["run_id"] = run_id
        usage_data["saved_at"] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(usage_data, f, indent=2)
        
        self.logger.info(f"Saved token usage to {filepath}")
        return filepath
    
    def save_question_usage(self, question_id: str, run_output_dir: str) -> str:
        """
        Save usage for a specific question.
        
        Returns:
            Path to saved file
        """
        question_usage = self.get_question_usage(question_id)
        
        filename = f"token_usage_q{question_id}.json"
        filepath = os.path.join(run_output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(question_usage, f, indent=2)
        
        return filepath
    
    def reset_session(self):
        """Reset session usage tracking."""
        with self._lock:
            self._session_usage.clear()
            self._total_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0
            }
        self.logger.info("Reset token counter session")
    
    def get_usage_summary(self) -> str:
        """Get a formatted summary of current usage."""
        with self._lock:
            total = self._total_usage
            return (f"Token Usage Summary:\n"
                   f"  Total API Calls: {total['api_calls']}\n"
                   f"  Input Tokens: {total['input_tokens']:,}\n"
                   f"  Output Tokens: {total['output_tokens']:,}\n"
                   f"  Total Tokens: {total['total_tokens']:,}")


# Global token counter instance
_global_token_counter = None
_counter_lock = threading.Lock()


def get_token_counter(output_dir: str = "token_logs", **kwargs) -> TokenCounter:
    """Get or create global token counter instance (thread-safe singleton)."""
    global _global_token_counter
    
    with _counter_lock:
        if _global_token_counter is None:
            _global_token_counter = TokenCounter(output_dir=output_dir, **kwargs)
    
    return _global_token_counter


def reset_global_counter():
    """Reset the global token counter."""
    global _global_token_counter
    
    with _counter_lock:
        if _global_token_counter:
            _global_token_counter.reset_session()


# Convenience functions
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Convenience function to count tokens in text."""
    counter = get_token_counter()
    return counter.count_tokens(text, model)


def count_message_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> int:
    """Convenience function to count tokens in messages."""
    counter = get_token_counter()
    return counter.count_message_tokens(messages, model)


def track_api_call(input_tokens: int, output_tokens: int, model: str, **kwargs) -> TokenUsage:
    """Convenience function to track an API call."""
    counter = get_token_counter()
    return counter.track_api_call(input_tokens, output_tokens, model, **kwargs)