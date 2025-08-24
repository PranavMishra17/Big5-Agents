"""
Base agent class for modular agent system with OpenAI API support.
"""

import logging
import json
import re
import time
import signal
import threading
from typing import List, Dict, Any, Optional, Tuple
import copy
from PIL import Image
import io
import base64

from openai import OpenAI
import config

from utils.prompts import (
    AGENT_SYSTEM_PROMPTS,
    LEADERSHIP_PROMPTS,
    COMMUNICATION_PROMPTS,
    MONITORING_PROMPTS,
    MENTAL_MODEL_PROMPTS,
    ORIENTATION_PROMPTS,
    TRUST_PROMPTS
)
from utils.token_counter import get_token_counter, TokenUsage


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


class Agent:
    """Base agent class for modular agent system with OpenAI API support."""
    
    def __init__(self, 
                 role: str, 
                 expertise_description: str,
                 use_team_leadership: bool = False,
                 use_closed_loop_comm: bool = False,
                 use_mutual_monitoring: bool = False,
                 use_shared_mental_model: bool = False,
                 use_team_orientation: bool = False,
                 use_mutual_trust: bool = False,
                 n_max: int = 5,
                 examples: Optional[List[Dict[str, str]]] = None,
                 deployment_config: Optional[Dict[str, str]] = None,
                 agent_index: int = 0,
                 task_config: Optional[Dict[str, Any]] = None):
        """
        Initialize an OpenAI-based agent with a specific role.
        
        Args:
            role: The role of the agent (e.g., "Critical Analyst")
            expertise_description: Description of the agent's expertise
            use_team_leadership: Whether this agent uses team leadership behaviors
            use_closed_loop_comm: Whether this agent uses closed-loop communication
            use_mutual_monitoring: Whether this agent uses mutual performance monitoring
            use_shared_mental_model: Whether this agent uses shared mental models
            use_team_orientation: Whether this agent uses team orientation
            use_mutual_trust: Whether this agent uses mutual trust
            n_max: Maximum number of agents
            examples: Optional examples to include in the prompt
            deployment_config: Optional specific deployment configuration
            agent_index: Index of agent for deployment assignment
        """
        self.role = role
        self.expertise_description = expertise_description
        self.use_team_leadership = use_team_leadership
        self.use_closed_loop_comm = use_closed_loop_comm
        self.use_mutual_monitoring = use_mutual_monitoring
        self.use_shared_mental_model = use_shared_mental_model
        self.use_team_orientation = use_team_orientation
        self.use_mutual_trust = use_mutual_trust
        self.n_max = n_max
        self.examples = examples or []
        self.conversation_history = []
        self.knowledge_base = {}
        self.agent_index = agent_index
        
        # Initialize context for token tracking
        self.question_id = None
        self.simulation_id = None
        
        # Initialize logger
        self.logger = logging.getLogger(f"agent.{role}")

        # Get deployment configuration
        if deployment_config:
            self.deployment_config = deployment_config
        else:
            self.deployment_config = config.get_deployment_for_agent(agent_index)
        
        # Check if this is a vision task
        self.is_vision_task = False
        if task_config and "image_data" in task_config:
            image_data = task_config["image_data"]
            self.is_vision_task = (
                image_data.get("requires_visual_analysis", False) and 
                image_data.get("image_available", False)
            )
        
        # Detect deployment type and initialize appropriate client
        self.deployment_type = self.deployment_config.get("type", "vertex_ai")
        
        if self.deployment_type == "vertex_ai":
            # Initialize Vertex AI using the simple approach
            from google.cloud import aiplatform
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.deployment_config["project"],
                location=self.deployment_config["location"]
            )
            
            # Get the endpoint using the simple approach
            self.endpoint = aiplatform.Endpoint(
                f"projects/{self.deployment_config['project']}/"
                f"locations/{self.deployment_config['location']}/"
                f"endpoints/{self.deployment_config['endpoint_id']}"
            )
            
            self.model = self.deployment_config["model"]
            
            self.logger.info(f"Initialized {self.role} with Vertex AI deployment {self.deployment_config['name']} using simple endpoint approach")
            
        else:
            # Fallback OpenAI initialization (should not be used in SLM branch)
            self.logger.warning(f"OpenAI deployment type detected in SLM branch - this should not happen")
            client_params = {
                "api_key": self.deployment_config["api_key"],
                "timeout": config.REQUEST_TIMEOUT
            }
            
            if self.deployment_config.get("organization"):
                client_params["organization"] = self.deployment_config["organization"]
            
            self.client = OpenAI(**client_params)
            self.model = self.deployment_config.get("model", "gpt-4o")
        
        # Build initial system message using global config (for backward compatibility)
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add example conversations if provided
        if self.examples:
            for example in self.examples:
                self.messages.append({"role": "user", "content": example['question']})
                self.messages.append({
                    "role": "assistant", 
                    "content": example['answer'] + "\n\n" + example.get('reason', '')
                })
                
        self.logger.info(f"Initialized {self.role} agent with deployment {self.deployment_config['name']} using model {self.model}")
    
    def _build_system_prompt(self, task_config: Dict[str, Any] = None) -> str:
        """Build the system prompt for the agent using isolated or global task config."""
        # Use provided task config or fall back to global
        task_config = task_config or config.TASK
        
        # Check if this is a pathology vision task
        is_pathology_vision = (
            self.is_vision_task and 
            task_config.get("image_data", {}).get("is_pathology_image", False)
        )
        
        # Choose appropriate prompt template
        if is_pathology_vision:
            prompt_template = "pathology_vision"
        else:
            prompt_template = "base"
            
        # Base prompt
        prompt = AGENT_SYSTEM_PROMPTS[prompt_template].format(
            role=self.role,
            expertise_description=self.expertise_description,
            team_name=config.TEAM_NAME,
            team_goal=config.TEAM_GOAL,
            task_name=task_config['name'],
            task_description=task_config['description'],
            task_type=task_config['type'],
            expected_output_format=task_config.get('expected_output_format', 'not specified')
        )

        # Add team components if enabled
        if self.use_team_leadership:
            prompt += LEADERSHIP_PROMPTS["team_leadership"]
        if self.use_closed_loop_comm:
            prompt += COMMUNICATION_PROMPTS["closed_loop"]
        if self.use_mutual_monitoring:
            prompt += MONITORING_PROMPTS["mutual_monitoring"]
        if self.use_shared_mental_model:
            prompt += MENTAL_MODEL_PROMPTS["shared_mental_model"]
        if self.use_team_orientation:
            prompt += ORIENTATION_PROMPTS["team_orientation"]
        if self.use_mutual_trust:
            prompt += TRUST_PROMPTS["mutual_trust_base"]
            trust_factor = getattr(self, 'mutual_trust_factor', 0.8)
            if trust_factor < 0.4:
                prompt += TRUST_PROMPTS["low_trust"]
            elif trust_factor > 0.7:
                prompt += TRUST_PROMPTS["high_trust"]
        
        return prompt

    def add_to_knowledge_base(self, key: str, value: Any) -> None:
        """Add information to the agent's knowledge base."""
        self.knowledge_base[key] = value
        self.logger.info(f"Added to knowledge base: {key}")
    
    def get_from_knowledge_base(self, key: str) -> Any:
        """Retrieve information from the agent's knowledge base."""
        return self.knowledge_base.get(key)
    
    def set_tracking_context(self, question_id: str = None, simulation_id: str = None):
        """Set context for token tracking."""
        self.question_id = question_id
        self.simulation_id = simulation_id
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        raise TimeoutError("Request timed out")
    

    def _encode_image_for_openai(self, image) -> Dict[str, Any]:
        """Encode image for OpenAI Vision API."""
        if image is None:
            return None
        
        try:
            # Process image
            img_copy = image.copy()
            if img_copy.mode != 'RGB':
                if img_copy.mode == 'RGBA':
                    background = Image.new('RGB', img_copy.size, (255, 255, 255))
                    background.paste(img_copy, mask=img_copy.split()[-1])
                    img_copy = background
                else:
                    img_copy = img_copy.convert('RGB')
            
            # Resize if needed  
            max_size = 1024
            if max(img_copy.size) > max_size:
                ratio = max_size / max(img_copy.size)
                new_size = tuple(int(dim * ratio) for dim in img_copy.size)
                img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)
            
            # Encode for OpenAI
            buffer = io.BytesIO()
            img_copy.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            image_bytes = buffer.read()
            buffer.close()
            
            # Validate we have data
            if len(image_bytes) == 0:
                self.logger.error("Empty image bytes")
                return None
            
            # Base64 encode
            base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Validate base64
            if len(base64_encoded_data) < 100:
                self.logger.error(f"Base64 too short: {len(base64_encoded_data)}")
                return None
            
            # Return OpenAI format
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_encoded_data}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image encoding failed: {str(e)}")
            return None

    def chat_with_image(self, message: str, image=None, max_tokens: Optional[int] = None) -> str:
        """Chat with image using appropriate API (Vision not fully supported for Vertex AI in SLM branch)."""
        if not getattr(self, 'is_vision_task', False):
            return self.chat(message)
        
        if image is None:
            return self.chat(message, max_tokens=max_tokens)
        
        # Apply MedRAG if available
        enhanced_message = message
        if self.get_from_knowledge_base("has_medrag_enhancement"):
            medrag_context = self.get_from_knowledge_base("medrag_context")
            if medrag_context:
                enhanced_message = f"{message}\n\n{medrag_context}\n\nIntegrate visual and literature findings."
        
        if self.deployment_type == "vertex_ai":
            # Vertex AI vision not fully implemented, use text-only fallback
            self.logger.warning("Vertex AI vision not fully implemented, using text-only fallback")
            return self.chat(f"{enhanced_message}\n\n[Image provided but vision analysis not available for Vertex AI models - describe what you would analyze based on the context]")
        else:
            # Existing OpenAI vision logic (fallback - should not be used in SLM branch)
            self.logger.warning("Using OpenAI vision fallback in SLM branch")
            
            # Encode image
            image_content = self._encode_image_for_openai(image)
            if not image_content:
                self.logger.error("Image encoding failed")
                return self.chat(f"{enhanced_message}\n\n[Image processing failed]")
            
            # Create message with image
            vision_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_message},
                    image_content
                ]
            }
            
            messages = self.messages + [vision_message]
            
            # API call with retries and enhanced token tracking
            for attempt in range(config.MAX_RETRIES):
                try:
                    # Get token counter and calculate tokens with vision support
                    token_counter = get_token_counter()
                    
                    # Enhanced token counting for vision messages
                    total_input_tokens, text_tokens, image_tokens = token_counter.count_message_tokens(messages, self.model)
                    
                    # Validate token limits
                    is_valid, token_message = token_counter.validate_token_limits(total_input_tokens, self.model)
                    if not is_valid:
                        self.logger.warning(f"Token limit validation failed: {token_message}")
                    
                    # Prepare API call parameters
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": config.TEMPERATURE,
                        "max_tokens": max_tokens if max_tokens else 4000  # Use passed param or default for vision
                    }
                    
                    # Time the API call
                    start_time = time.time()
                    response = self.client.chat.completions.create(**api_params)
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    
                    assistant_message = response.choices[0].message.content
                    
                    # Count output tokens
                    output_tokens = token_counter.count_tokens(assistant_message, self.model)
                    
                    # Track the API call with enhanced vision token data
                    token_counter.track_api_call(
                        input_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        model=self.model,
                        agent_role=self.role,
                        question_id=self.question_id,
                        simulation_id=self.simulation_id,
                        operation_type="chat_with_image",
                        response_time_ms=response_time_ms,
                        image_tokens=image_tokens,
                        text_tokens=text_tokens,
                        num_images=1
                    )
                    
                    # Store in conversation (avoid corrupted image data)
                    self.messages.append({
                        "role": "user", 
                        "content": message  # Store as simple text
                    })
                    self.messages.append({"role": "assistant", "content": assistant_message})
                    
                    self.conversation_history.append({
                        "user": message,
                        "assistant": assistant_message,
                        "has_image": True,
                        "image_tokens": image_tokens,
                        "text_tokens": text_tokens
                    })
                    
                    self.logger.debug(f"Vision API call completed - Input: {total_input_tokens} ({text_tokens} text + {image_tokens} image), "
                                    f"Output: {output_tokens}, Total: {total_input_tokens + output_tokens} tokens, "
                                    f"Time: {response_time_ms:.1f}ms")
                    
                    return assistant_message
                    
                except Exception as e:
                    self.logger.warning(f"Vision attempt {attempt + 1} failed: {str(e)}")
                    
                    if "image" in str(e).lower() or "vision" in str(e).lower():
                        self.logger.error("Vision error, falling back to text")
                        return self.chat(f"{enhanced_message}\n\n[Image processing failed]")
                    
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(config.RETRY_DELAY * (2 ** attempt))
                    else:
                        return self.chat(f"{enhanced_message}\n\n[Vision failed after retries]", max_tokens=max_tokens)


    def chat(self, message: str, max_tokens: Optional[int] = None) -> str:
        """Send a message to the agent and get a response using appropriate API (Vertex AI for SLM branch)."""
        self.logger.info(f"Received message: {message[:100]}...")
        
        # Apply MedRAG enhancement if available
        enhanced_message = message
        if self.get_from_knowledge_base("has_medrag_enhancement"):
            medrag_context = self.get_from_knowledge_base("medrag_context")
            if medrag_context:
                enhanced_message = f"""{message}

{medrag_context}

Based on the retrieved medical literature above, provide your analysis considering this evidence alongside your clinical expertise. If the literature supports, contradicts, or adds important context to your reasoning, please integrate this into your response."""
                
            self.logger.info("Applied MedRAG knowledge enhancement to agent message")
        
        for attempt in range(config.MAX_RETRIES):
            try:
                start_time = time.time()
                
                # Route to appropriate API
                if self.deployment_type == "vertex_ai":
                    assistant_message = self._chat_vertex_ai(enhanced_message)
                    # Approximate token counting for Vertex AI
                    input_tokens = len(enhanced_message.split()) * 1.3
                    output_tokens = len(assistant_message.split()) * 1.3
                    
                else:
                    # Existing OpenAI logic (fallback - should not be used in SLM branch)
                    self.logger.warning("Using OpenAI fallback in SLM branch")
                    user_message = {"role": "user", "content": enhanced_message}
                    messages = self.messages + [user_message]
                    
                    # Get token counter and calculate tokens with enhanced vision support
                    token_counter = get_token_counter()
                    total_input_tokens, text_tokens, image_tokens = token_counter.count_message_tokens(messages, self.model)
                    
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": config.TEMPERATURE
                    }
                    
                    if max_tokens:
                        api_params["max_tokens"] = max_tokens
                    elif config.MAX_TOKENS:
                        api_params["max_tokens"] = config.MAX_TOKENS
                    
                    response = self.client.chat.completions.create(**api_params)
                    assistant_message = response.choices[0].message.content
                    
                    # Accurate token counting for OpenAI
                    output_tokens = token_counter.count_tokens(assistant_message, self.model)
                    input_tokens = total_input_tokens
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Track API call with unified interface
                get_token_counter().track_api_call(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    model=self.model,
                    agent_role=self.role,
                    question_id=self.question_id,
                    simulation_id=self.simulation_id,
                    operation_type=f"{self.deployment_type}_chat",
                    response_time_ms=response_time_ms
                )
                
                # Store response (use original message, not enhanced)
                self.messages.append({"role": "user", "content": message})
                self.messages.append({"role": "assistant", "content": assistant_message})
                self.conversation_history.append({
                    "user": message,
                    "assistant": assistant_message
                })
                
                self.logger.debug(f"Chat completed via {self.deployment_type} - Input: {int(input_tokens)}, Output: {int(output_tokens)}, Time: {response_time_ms:.1f}ms")
                return assistant_message
                
            except Exception as e:
                self.logger.warning(f"Chat attempt {attempt + 1} failed: {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY * (2 ** attempt))
                else:
                    raise Exception(f"All attempts failed: {str(e)}")


    def _chat_with_timeout(self, messages: List[Dict[str, Any]], timeout: int = None) -> str:
        """Chat with timeout and retry logic for OpenAI API with enhanced token tracking."""
        if timeout is None:
            timeout = config.INACTIVITY_TIMEOUT
        
        def target():
            try:
                # Get token counter and calculate tokens with enhanced vision support
                token_counter = get_token_counter()
                
                # Enhanced token counting for all message types
                total_input_tokens, text_tokens, image_tokens = token_counter.count_message_tokens(messages, self.model)
                
                # Validate token limits
                is_valid, message = token_counter.validate_token_limits(total_input_tokens, self.model)
                if not is_valid:
                    self.logger.warning(f"Token limit validation failed: {message}")
                
                # Check if this is a vision call
                has_image = image_tokens > 0
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": config.TEMPERATURE
                }
                
                # Add max_tokens for vision calls or if specified
                max_tokens = None
                if has_image or self.is_vision_task:
                    max_tokens = 4000
                    api_params["max_tokens"] = max_tokens
                elif config.MAX_TOKENS:
                    max_tokens = config.MAX_TOKENS
                    api_params["max_tokens"] = max_tokens
                
                # Time the API call
                start_time = time.time()
                response = self.client.chat.completions.create(**api_params)
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                self._response = response.choices[0].message.content
                
                # Count output tokens
                output_tokens = token_counter.count_tokens(self._response, self.model)
                
                # Track the API call with enhanced timing and vision data
                token_counter.track_api_call(
                    input_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    agent_role=self.role,
                    question_id=self.question_id,
                    simulation_id=self.simulation_id,
                    operation_type="chat_with_timeout",
                    response_time_ms=response_time_ms,
                    image_tokens=image_tokens if has_image else None,
                    text_tokens=text_tokens if has_image else total_input_tokens,
                    num_images=self._count_images_in_messages(messages) if has_image else None
                )
                
                log_msg = f"API call completed - Input: {total_input_tokens}, Output: {output_tokens}, Total: {total_input_tokens + output_tokens} tokens"
                if has_image:
                    log_msg += f" (Vision: {text_tokens} text + {image_tokens} image)"
                log_msg += f", Time: {response_time_ms:.1f}ms"
                self.logger.debug(log_msg)
                    
            except Exception as e:
                self.logger.error(f"OpenAI API call error: {str(e)}")
                self._exception = e
        
        # Reset response and exception
        self._response = None
        self._exception = None
        
        # Start thread for API call
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            self.logger.warning(f"Request timed out after {timeout} seconds")
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        
        if self._exception:
            raise self._exception
        
        if self._response is None:
            raise Exception("No response received")
        
        return self._response

    def _chat_vertex_ai(self, message: str) -> str:
        """Chat using Vertex AI deployed model with simple endpoint approach."""
        try:
            # Prepare instance for Vertex AI prediction using simple format
            instances = [{"prompt": message}]
            
            # Add parameters if needed
            parameters = {
                "temperature": config.TEMPERATURE,
                "max_output_tokens": config.MAX_TOKENS or 1500,
                "top_p": 0.8,
                "top_k": 40
            }
            
            self.logger.debug(f"Making prediction with instances: {instances}")
            
            # Make prediction using the simple endpoint approach
            response = self.endpoint.predict(instances=instances, parameters=parameters)
            
            self.logger.debug(f"Raw prediction response: {response}")
            
            # Parse response - try different ways to extract the content
            if response.predictions:
                prediction = response.predictions[0]
                self.logger.debug(f"First prediction: {prediction}")
                
                # Handle different response formats
                if isinstance(prediction, dict):
                    # Try different response fields
                    content = (prediction.get("content") or
                              prediction.get("generated_text") or 
                              prediction.get("output") or
                              prediction.get("response") or
                              prediction.get("text") or
                              str(prediction))
                elif isinstance(prediction, str):
                    content = prediction
                    
                    # If the response contains "Output:" extract what comes after it
                    if "Output:" in content:
                        output_part = content.split("Output:", 1)[1].strip()
                        if output_part:
                            content = output_part
                else:
                    # Convert to string if it's some other type
                    content = str(prediction)
                
                return content if content and content != str(prediction) else "No response generated"
            
            return "No response generated"
            
        except Exception as e:
            self.logger.error(f"Vertex AI prediction failed: {str(e)}")
            raise

    def _count_images_in_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count the number of images in a list of messages."""
        image_count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_count += 1
        return image_count

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    # Keep all the existing extraction methods unchanged
    def extract_response(self, message=None) -> Dict[str, Any]:
        """Extract the agent's response to the task from their message using global config."""
        if message is None:
            if not self.conversation_history:
                return {}
            message = self.conversation_history[-1]["assistant"]
        
        return self.extract_response_isolated(message, config.TASK)
    
    def extract_response_isolated(self, message: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the agent's response to the task from their message using isolated task config."""
        task_type = task_config["type"]
        
        if task_type == "ranking":
            return self.extract_ranking_isolated(message, task_config)
        elif task_type == "mcq":
            return self.extract_mcq_answer_isolated(message, task_config)
        elif task_type == "multi_choice_mcq":
            return self.extract_multi_choice_mcq_answer_isolated(message, task_config)
        elif task_type == "yes_no_maybe":
            return self.extract_yes_no_maybe_answer(message)
        elif task_type in ["open_ended", "estimation", "selection"]:
            return {"response": message, "confidence": self.extract_confidence(message)}
        else:
            return {"raw_response": message}

    def extract_ranking_isolated(self, message: str, task_config: Dict[str, Any]):
        """Extract a ranking from the agent's response using isolated task config."""
        ranking = []
        lines = message.split('\n')
        
        # Look for numbered items (1. Item, 2. Item, etc.)
        for line in lines:
            for i in range(1, len(task_config["options"]) + 1):
                if f"{i}." in line or f"{i}:" in line:
                    for item in task_config["options"]:
                        if item.lower() in line.lower():
                            ranking.append(item)
                            break
        
        # Check for duplicates and missing items
        seen_items = set()
        valid_ranking = []
        
        for item in ranking:
            if item not in seen_items:
                seen_items.add(item)
                valid_ranking.append(item)
        
        # Add any missing items at the end
        for item in task_config["options"]:
            if item not in seen_items:
                valid_ranking.append(item)
                seen_items.add(item)
        
        # Extract confidence level
        confidence = self.extract_confidence(message)
        
        return {
            "ranking": valid_ranking[:len(task_config["options"])],
            "confidence": confidence
        }
    
    def extract_mcq_answer_isolated(self, message: str, task_config: Dict[str, Any]):
        """Extract an MCQ answer from the agent's response using isolated task config."""
        # Look for option identifiers (A, B, C, D, etc.)
        for line in message.split('\n'):
            line = line.strip()
            for option in task_config["options"]:
                option_id = option.split('.')[0].strip() if '.' in option else None
                if option_id and (line.startswith(option_id) or f"Option {option_id}" in line or f"Answer: {option_id}" in line):
                    confidence = self.extract_confidence(message)
                    return {"answer": option_id, "confidence": confidence}
        
        confidence = self.extract_confidence(message)
        
        # If no explicit option identifier found, look for full option text
        for line in message.split('\n'):
            for option in task_config["options"]:
                option_id = option.split('.')[0].strip() if '.' in option else None
                if option[2:].strip().lower() in line.lower():
                    return {"answer": option_id, "confidence": confidence}
        
        # Look for selection indicators
        lower_message = message.lower()
        for option in task_config["options"]:
            option_id = option.split('.')[0].strip() if '.' in option else None
            if option_id and f"select {option_id.lower()}" in lower_message or f"choose {option_id.lower()}" in lower_message:
                return {"answer": option_id, "confidence": confidence}
        
        return {"answer": None, "confidence": confidence}
    
    def extract_multi_choice_mcq_answer_isolated(self, message: str, task_config: Dict[str, Any]):
        """Extract multi-choice MCQ answers from the agent's response using isolated task config."""
        import re
        
        patterns = [
            r"ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"FINAL ANSWERS?:\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"selected options?:?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)",
            r"correct options? (?:are|is):?\s*([A-D],?\s*(?:and\s*)?[A-D]?(?:,?\s*(?:and\s*)?[A-D])?(?:,?\s*(?:and\s*)?[A-D])?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                answer_str = match.group(1).upper()
                answer_str = answer_str.replace('AND', ',').replace(' ', '')
                answers = [a.strip() for a in answer_str.split(',') if a.strip()]
                answers = sorted(list(set(answers)))
                confidence = self.extract_confidence(message)
                return {"answers": answers, "confidence": confidence}
        
        option_pattern = r"options?\s+([A-D](?:\s*,\s*[A-D])*(?:\s*,?\s*and\s*[A-D])?)"
        match = re.search(option_pattern, message, re.IGNORECASE)
        if match:
            answer_str = match.group(1).upper()
            answer_str = answer_str.replace('AND', ',').replace(' ', '')
            answers = [a.strip() for a in answer_str.split(',') if a.strip()]
            answers = sorted(list(set(answers)))
            confidence = self.extract_confidence(message)
            return {"answers": answers, "confidence": confidence}
        
        confidence = self.extract_confidence(message)
        return {"answers": [], "confidence": confidence}

    def extract_yes_no_maybe_answer(self, content):
        """Extract yes/no/maybe answer with improved parsing."""
        if not isinstance(content, str):
            return None
        
        content_lower = content.lower()
        
        patterns = [
            r"ANSWER:\s*(yes|no|maybe)",
            r"^ANSWER:\s*(yes|no|maybe)",
            r"FINAL ANSWER:\s*(yes|no|maybe)",
            r"answer is:?\s*(yes|no|maybe)",
            r"my answer:?\s*(yes|no|maybe)",
            r"the answer is:?\s*(yes|no|maybe)",
            r"conclusion:?\s*(yes|no|maybe)",
            r"therefore:?\s*(yes|no|maybe)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).lower()
        
        lines = content_lower.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('yes,') or line.startswith('yes.') or line == 'yes':
                return "yes"
            elif line.startswith('no,') or line.startswith('no.') or line == 'no':
                return "no"
            elif line.startswith('maybe') or line.startswith('uncertain') or line.startswith('possibly'):
                return "maybe"
        
        return None
    
    def extract_confidence(self, message):
        """Extract a confidence level from the agent's response."""
        confidence = 0.7
        lower_message = message.lower()
        
        if "confidence: " in lower_message:
            confidence_parts = lower_message.split("confidence: ")[1].split()
            if confidence_parts:
                try:
                    if "%" in confidence_parts[0]:
                        confidence_value = float(confidence_parts[0].replace("%", "")) / 100
                        confidence = min(max(confidence_value, 0.0), 1.0)
                    else:
                        confidence_value = float(confidence_parts[0])
                        if confidence_value > 1.0:
                            confidence_value /= 10
                        confidence = min(max(confidence_value, 0.0), 1.0)
                except ValueError:
                    pass
        
        if confidence == 0.7:
            high_confidence_markers = ["certainly", "definitely", "absolutely", "strongly believe", "confident", "sure", "clear evidence", "conclusive"]
            medium_confidence_markers = ["likely", "probably", "think", "believe", "reasonable", "should be", "suggests", "indicates"]
            low_confidence_markers = ["uncertain", "might", "possibly", "guess", "not sure", "doubtful", "maybe", "unclear", "insufficient"]
            
            high_count = sum(1 for marker in high_confidence_markers if marker in lower_message)
            medium_count = sum(1 for marker in medium_confidence_markers if marker in lower_message)
            low_count = sum(1 for marker in low_confidence_markers if marker in lower_message)
            
            if high_count > medium_count and high_count > low_count:
                confidence = 0.9
            elif medium_count > low_count:
                confidence = 0.7
            elif low_count > 0:
                confidence = 0.4
            
        return confidence

    def share_knowledge(self, other_agent):
        """Share knowledge with another agent."""
        shared_knowledge = {}
        
        if "task_understanding" in self.knowledge_base:
            shared_knowledge["task_understanding"] = self.knowledge_base["task_understanding"]
            other_agent.add_to_knowledge_base("task_understanding", self.knowledge_base["task_understanding"])
        
        if "reasoning_approaches" in self.knowledge_base:
            shared_knowledge["reasoning_approaches"] = self.knowledge_base["reasoning_approaches"]
            other_agent.add_to_knowledge_base("reasoning_approaches", self.knowledge_base["reasoning_approaches"])
        
        self.logger.info(f"Agent {self.role} shared knowledge with {other_agent.role}")
        return shared_knowledge