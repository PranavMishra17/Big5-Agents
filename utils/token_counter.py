"""
Enhanced token counting utility with proper vision token calculation.
Integrates seamlessly with existing TokenCounter class.
"""
import tiktoken
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from math import ceil
from PIL import Image
import io
import base64


@dataclass
class TokenUsage:
    """Enhanced data class for tracking token usage including vision tokens."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: str
    agent_role: Optional[str] = None
    question_id: Optional[str] = None
    simulation_id: Optional[str] = None
    operation_type: Optional[str] = None
    response_time_ms: Optional[float] = None
    # New vision-specific fields
    image_tokens: Optional[int] = None
    text_tokens: Optional[int] = None
    num_images: Optional[int] = None
    vision_call: Optional[bool] = None


class EnhancedTokenCounter:
    """
    Enhanced token counter with proper vision token calculation.
    Replaces the existing TokenCounter with full backward compatibility.
    """
    
    def __init__(self, output_dir: str = "token_logs", max_input_tokens: int = 100000, max_output_tokens: int = 8192):
        """Initialize enhanced token counter with vision support."""
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
            "api_calls": 0,
            "image_tokens": 0,  # New: track image tokens separately
            "text_tokens": 0,   # New: track text tokens separately
            "vision_calls": 0   # New: track vision API calls
        }
        
        # Time tracking
        self._session_start_time = datetime.now()
        self._total_response_time_ms = 0.0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize encoders
        self._encoders = {}
        self._initialize_encoders()
        
    def _initialize_encoders(self):
        """Initialize tiktoken encoders for common models."""
        try:
            self._encoders["gpt-4"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-4o"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-4-turbo"] = tiktoken.get_encoding("cl100k_base")
            self._encoders["gpt-3.5-turbo"] = tiktoken.get_encoding("cl100k_base")
            self.logger.info("Initialized tiktoken encoders for vision token counting")
        except Exception as e:
            self.logger.error(f"Failed to initialize tiktoken encoders: {e}")
            
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get the appropriate encoder for a model."""
        model_key = model.lower()
        if "gpt-4o" in model_key:
            return self._encoders.get("gpt-4o")
        elif "gpt-4" in model_key:
            return self._encoders.get("gpt-4")
        elif "gpt-3.5" in model_key:
            return self._encoders.get("gpt-3.5-turbo")
        else:
            return self._encoders.get("gpt-4", tiktoken.get_encoding("cl100k_base"))
    
    def calculate_image_tokens(self, image, detail: str = "auto") -> int:
        """
        Calculate vision tokens for an image based on OpenAI's formula.
        
        Args:
            image: PIL Image object or image dimensions (width, height)
            detail: "low", "high", or "auto"
            
        Returns:
            Number of tokens required for this image
        """
        try:
            # Handle different image input types
            if isinstance(image, tuple):
                width, height = image
            elif hasattr(image, 'size'):
                width, height = image.size
            else:
                # Fallback: assume standard image size
                self.logger.warning("Could not determine image dimensions, using default estimate")
                return 765  # Common medium-sized image token count
            
            # For "low" detail, always return base tokens
            if detail == "low":
                return 85
            
            # For "high" detail or "auto", calculate based on tiles
            return self._calculate_high_detail_tokens(width, height)
            
        except Exception as e:
            self.logger.warning(f"Image token calculation failed: {e}")
            return 765  # Safe default estimate
    
    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """
        Calculate tokens for high-detail image processing.
        Based on OpenAI's official formula.
        """
        # Step 1: Scale to fit within 2048x2048 (maintain aspect ratio)
        if width > 2048 or height > 2048:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                width, height = 2048, int(2048 / aspect_ratio)
            else:
                width, height = int(2048 * aspect_ratio), 2048
        
        # Step 2: Scale so shortest side is 768px
        if width >= height and height > 768:
            width, height = int((768 / height) * width), 768
        elif height > width and width > 768:
            width, height = 768, int((768 / width) * height)
        
        # Step 3: Calculate number of 512x512 tiles
        tiles_width = ceil(width / 512)
        tiles_height = ceil(height / 512)
        num_tiles = tiles_width * tiles_height
        
        # Step 4: Apply OpenAI formula
        # Base tokens (85) + tile tokens (170 per tile)
        total_tokens = 85 + 170 * num_tiles
        
        self.logger.debug(f"Image tokens: {width}x{height} -> {num_tiles} tiles -> {total_tokens} tokens")
        return total_tokens
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text using appropriate encoder."""
        try:
            encoder = self.get_encoder(model)
            if encoder:
                return len(encoder.encode(text))
            else:
                return len(text) // 4  # Fallback estimation
        except Exception as e:
            self.logger.warning(f"Token counting failed, using estimation: {e}")
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, Any]], model: str = "gpt-4o") -> Tuple[int, int, int]:
        """
        Enhanced message token counting with proper vision support.
        
        Returns:
            Tuple of (total_tokens, text_tokens, image_tokens)
        """
        try:
            encoder = self.get_encoder(model)
            text_tokens = 0
            image_tokens = 0
            
            for message in messages:
                # Base message formatting overhead
                text_tokens += 4  # Message structure tokens
                
                content = message.get("content")
                if isinstance(content, str):
                    # Simple text message
                    text_tokens += len(encoder.encode(content))
                    
                elif isinstance(content, list):
                    # Vision message with mixed content
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_tokens += len(encoder.encode(item.get("text", "")))
                                
                            elif item.get("type") == "image_url":
                                # Calculate image tokens
                                image_url_data = item.get("image_url", {})
                                detail = image_url_data.get("detail", "auto")
                                
                                # Try to get image dimensions for accurate calculation
                                url = image_url_data.get("url", "")
                                if url.startswith("data:image"):
                                    # Base64 encoded image - try to decode and get dimensions
                                    try:
                                        header, data = url.split(",", 1)
                                        image_data = base64.b64decode(data)
                                        with Image.open(io.BytesIO(image_data)) as img:
                                            image_tokens += self.calculate_image_tokens(img, detail)
                                    except Exception:
                                        # Fallback to standard estimate
                                        image_tokens += 765 if detail != "low" else 85
                                else:
                                    # External URL - use standard estimate based on detail
                                    image_tokens += 765 if detail != "low" else 85
                
                # Handle other message fields (role, name, etc.)
                for key, value in message.items():
                    if key != "content" and isinstance(value, str):
                        text_tokens += len(encoder.encode(value))
            
            text_tokens += 2  # Assistant priming tokens
            total_tokens = text_tokens + image_tokens
            
            self.logger.debug(f"Message tokens: {total_tokens} total ({text_tokens} text + {image_tokens} image)")
            return total_tokens, text_tokens, image_tokens
            
        except Exception as e:
            self.logger.warning(f"Enhanced message token counting failed: {e}")
            # Fallback to simple estimation
            total_text = ""
            image_count = 0
            
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    total_text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                total_text += item.get("text", "")
                            elif item.get("type") == "image_url":
                                image_count += 1
            
            text_tokens = len(total_text) // 4 + len(messages) * 10
            image_tokens = image_count * 765  # Conservative estimate
            return text_tokens + image_tokens, text_tokens, image_tokens
    
    def validate_token_limits(self, input_tokens: int, model: str = "gpt-4o") -> Tuple[bool, str]:
        """Validate if token count is within limits."""
        if input_tokens > self.max_input_tokens:
            return False, f"Input tokens ({input_tokens}) exceed limit ({self.max_input_tokens})"
        
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385
        }
        
        model_key = model.lower()
        for key, limit in model_limits.items():
            if key in model_key:
                if input_tokens > limit * 0.8:
                    return False, f"Input tokens ({input_tokens}) too close to model limit ({limit})"
                break
        
        return True, "Token count within limits"
    
    def track_api_call(self, input_tokens: int, output_tokens: int, model: str, 
                      agent_role: str = None, question_id: str = None, 
                      simulation_id: str = None, operation_type: str = None,
                      response_time_ms: float = None, 
                      image_tokens: int = None, text_tokens: int = None,
                      num_images: int = None) -> TokenUsage:
        """
        Enhanced API call tracking with vision token support.
        
        Args:
            image_tokens: Number of tokens used for images
            text_tokens: Number of tokens used for text
            num_images: Number of images in the request
        """
        total_tokens = input_tokens + output_tokens
        is_vision_call = image_tokens is not None and image_tokens > 0
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            timestamp=datetime.now().isoformat(),
            agent_role=agent_role,
            question_id=question_id,
            simulation_id=simulation_id,
            operation_type=operation_type,
            response_time_ms=response_time_ms,
            image_tokens=image_tokens or 0,
            text_tokens=text_tokens or input_tokens,
            num_images=num_images or 0,
            vision_call=is_vision_call
        )
        
        with self._lock:
            self._session_usage.append(usage)
            self._total_usage["input_tokens"] += input_tokens
            self._total_usage["output_tokens"] += output_tokens
            self._total_usage["total_tokens"] += total_tokens
            self._total_usage["api_calls"] += 1
            self._total_usage["image_tokens"] += image_tokens or 0
            self._total_usage["text_tokens"] += text_tokens or input_tokens
            if is_vision_call:
                self._total_usage["vision_calls"] += 1
            if response_time_ms:
                self._total_response_time_ms += response_time_ms
        
        log_msg = f"Tracked API call: {input_tokens} in, {output_tokens} out, {total_tokens} total"
        if is_vision_call:
            log_msg += f" (Vision: {image_tokens} image + {text_tokens} text tokens, {num_images} images)"
        self.logger.debug(log_msg)
        
        return usage
    
    def get_session_usage(self) -> Dict[str, Any]:
        """Get enhanced session usage statistics with vision breakdown."""
        with self._lock:
            session_duration = (datetime.now() - self._session_start_time).total_seconds()
            avg_response_time = self._total_response_time_ms / max(self._total_usage["api_calls"], 1)
            
            # Calculate vision vs text call breakdown
            vision_calls = self._total_usage["vision_calls"]
            text_only_calls = self._total_usage["api_calls"] - vision_calls
            
            return {
                "total_usage": self._total_usage.copy(),
                "vision_breakdown": {
                    "total_image_tokens": self._total_usage["image_tokens"],
                    "total_text_tokens": self._total_usage["text_tokens"],
                    "vision_calls": vision_calls,
                    "text_only_calls": text_only_calls,
                    "vision_call_percentage": (vision_calls / max(self._total_usage["api_calls"], 1)) * 100
                },
                "timing_summary": {
                    "total_time_seconds": round(session_duration, 2),
                    "total_time_minutes": round(session_duration / 60, 2),
                    "average_time_per_call_ms": round(avg_response_time, 2),
                    "average_time_per_call_seconds": round(avg_response_time / 1000, 2),
                    "total_api_response_time_ms": round(self._total_response_time_ms, 2),
                    "session_start_time": self._session_start_time.isoformat(),
                    "session_end_time": datetime.now().isoformat()
                },
                "call_count": len(self._session_usage),
                "detailed_calls": [asdict(usage) for usage in self._session_usage]
            }
    
    def get_vision_statistics(self) -> Dict[str, Any]:
        """Get detailed vision usage statistics."""
        with self._lock:
            vision_calls = [usage for usage in self._session_usage if usage.vision_call]
            
            if not vision_calls:
                return {"no_vision_calls": True}
            
            total_images = sum(call.num_images or 0 for call in vision_calls)
            total_image_tokens = sum(call.image_tokens or 0 for call in vision_calls)
            avg_tokens_per_image = total_image_tokens / max(total_images, 1)
            
            return {
                "vision_calls": len(vision_calls),
                "total_images_processed": total_images,
                "total_image_tokens": total_image_tokens,
                "average_tokens_per_image": round(avg_tokens_per_image, 1),
                "vision_call_percentage": (len(vision_calls) / len(self._session_usage)) * 100,
                "image_token_percentage": (total_image_tokens / max(self._total_usage["total_tokens"], 1)) * 100
            }
    
    # All other methods remain the same as original TokenCounter
    def get_question_usage(self, question_id: str) -> Dict[str, Any]:
        """Get token usage for a specific question with vision breakdown."""
        with self._lock:
            question_calls = [usage for usage in self._session_usage if usage.question_id == question_id]
            
            if not question_calls:
                return {"error": f"No usage found for question {question_id}"}
            
            total_input = sum(call.input_tokens for call in question_calls)
            total_output = sum(call.output_tokens for call in question_calls)
            total_image_tokens = sum(call.image_tokens or 0 for call in question_calls)
            total_text_tokens = sum(call.text_tokens or 0 for call in question_calls)
            vision_calls = sum(1 for call in question_calls if call.vision_call)
            total_images = sum(call.num_images or 0 for call in question_calls)
            
            # Timing calculations
            question_response_times = [call.response_time_ms for call in question_calls if call.response_time_ms is not None]
            total_response_time = sum(question_response_times)
            avg_response_time = total_response_time / max(len(question_response_times), 1)
            
            return {
                "question_id": question_id,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "api_calls": len(question_calls),
                "vision_breakdown": {
                    "image_tokens": total_image_tokens,
                    "text_tokens": total_text_tokens,
                    "vision_calls": vision_calls,
                    "total_images": total_images,
                    "average_tokens_per_image": total_image_tokens / max(total_images, 1) if total_images > 0 else 0
                },
                "timing_summary": {
                    "total_api_response_time_ms": round(total_response_time, 2),
                    "average_response_time_ms": round(avg_response_time, 2),
                },
                "detailed_calls": [asdict(usage) for usage in question_calls]
            }
    
    def save_session_usage(self, run_id: str = None) -> str:
        """Save enhanced session usage with vision statistics."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"token_usage_{run_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        usage_data = self.get_session_usage()
        usage_data["run_id"] = run_id
        usage_data["saved_at"] = datetime.now().isoformat()
        usage_data["vision_statistics"] = self.get_vision_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(usage_data, f, indent=2)
        
        self.logger.info(f"Saved enhanced token usage to {filepath}")
        return filepath
    
    def reset_session(self):
        """Reset session usage tracking."""
        with self._lock:
            self._session_usage.clear()
            self._total_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
                "image_tokens": 0,
                "text_tokens": 0,
                "vision_calls": 0
            }
            self._total_response_time_ms = 0.0
        self.logger.info("Reset enhanced token counter session")
    
    def get_usage_summary(self) -> str:
        """Get a formatted summary of current usage including vision stats."""
        with self._lock:
            total = self._total_usage
            vision_stats = self.get_vision_statistics()
            
            summary = (f"Enhanced Token Usage Summary:\n"
                      f"  Total API Calls: {total['api_calls']} ({total['vision_calls']} vision, {total['api_calls'] - total['vision_calls']} text-only)\n"
                      f"  Input Tokens: {total['input_tokens']:,} ({total['text_tokens']:,} text + {total['image_tokens']:,} image)\n"
                      f"  Output Tokens: {total['output_tokens']:,}\n"
                      f"  Total Tokens: {total['total_tokens']:,}")
            
            if not vision_stats.get("no_vision_calls"):
                summary += (f"\n  Vision Statistics:\n"
                           f"    Images Processed: {vision_stats.get('total_images_processed', 0)}\n"
                           f"    Avg Tokens/Image: {vision_stats.get('average_tokens_per_image', 0):.1f}\n"
                           f"    Image Token %: {vision_stats.get('image_token_percentage', 0):.1f}%")
            
            return summary


# Updated global instance management
_global_token_counter = None
_counter_lock = threading.Lock()


def get_token_counter(output_dir: str = "token_logs", **kwargs):
    """Get or create enhanced global token counter instance."""
    global _global_token_counter
    
    with _counter_lock:
        if _global_token_counter is None:
            _global_token_counter = EnhancedTokenCounter(output_dir=output_dir, **kwargs)
    
    return _global_token_counter


def reset_global_counter():
    """Reset the global token counter."""
    global _global_token_counter
    
    with _counter_lock:
        if _global_token_counter:
            _global_token_counter.reset_session()


# Enhanced convenience functions
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Convenience function to count tokens in text."""
    counter = get_token_counter()
    return counter.count_tokens(text, model)


def count_message_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> Tuple[int, int, int]:
    """Enhanced convenience function to count tokens in messages with vision support."""
    counter = get_token_counter()
    return counter.count_message_tokens(messages, model)


def track_api_call(input_tokens: int, output_tokens: int, model: str, **kwargs) -> TokenUsage:
    """Enhanced convenience function to track an API call."""
    counter = get_token_counter()
    return counter.track_api_call(input_tokens, output_tokens, model, **kwargs)