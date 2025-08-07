"""
Closed-loop communication implementation for agent system.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional

from components.agent import Agent

from utils.prompts import COMMUNICATION_PROMPTS


class ClosedLoopCommunication:
    """
    Implements closed-loop communication protocol between agents.
    
    Closed-loop communication involves:
    1. Sender initiating a message
    2. Receiver acknowledging and confirming understanding
    3. Sender verifying the message was received correctly
    """
    
    def __init__(self):
        """Initialize the closed-loop communication handler."""
        self.logger = logging.getLogger("communication.closed_loop")
        self.logger.info("Initialized closed-loop communication handler")
        
        # Track communication metrics
        self.exchanges = []
        self.misunderstandings = 0
        self.clarifications = 0
    
    def facilitate_exchange(self, 
                           sender: Agent, 
                           receiver: Agent, 
                           initial_message: str) -> Tuple[str, str, str]:
        """
        STREAMLINED: Facilitate closed-loop communication with minimal API calls.
        
        Args:
            sender: The agent sending the initial message
            receiver: The agent receiving the message
            initial_message: The content of the initial message
            
        Returns:
            Tuple containing (initial message, acknowledgment, verification)
        """
        self.logger.info(f"Streamlined closed-loop: {sender.role} -> {receiver.role}")
        
        # STREAMLINED: Combined exchange in just 2 API calls instead of 3
        # Step 1: Sender sends message with closed-loop instruction
        sender_prompt = f"{initial_message}\n\nProvide your analysis. Be precise, concise, and to the point."
        sender_message = sender.chat(sender_prompt)
        self.logger.info(f"Sender analysis: {sender_message[:50]}...")
        
        # Step 2: Receiver responds with acknowledgment AND content
        receiver_prompt = f"""Message from {sender.role}: {sender_message}

Acknowledge receipt and provide your response. Format:
- "Understood: [key point]"  
- Your analysis/feedback

Be precise, concise, and to the point."""
        
        receiver_message = receiver.chat(receiver_prompt)
        self.logger.info(f"Receiver response: {receiver_message[:50]}...")
        
        # SIMPLIFIED: Skip verification step to save API call - assume understanding
        verification_message = "Streamlined exchange completed"
        
        # Track simplified exchange
        exchange_data = {
            "sender_role": sender.role,
            "receiver_role": receiver.role,
            "initial_message": sender_message,
            "acknowledgment": receiver_message,
            "verification": verification_message,
            "streamlined": True
        }
        
        self.exchanges.append(exchange_data)
        
        return (sender_message, receiver_message, verification_message)
    
    def _detect_misunderstanding(self, verification_message: str) -> bool:
        """Detect if there was a misunderstanding in the exchange."""
        misunderstanding_indicators = [
            "misunderstood", "misunderstanding", "didn't understand", "did not understand", 
            "missed", "incorrect", "not quite", "mistaken", "misinterpreted"
        ]
        
        for indicator in misunderstanding_indicators:
            if indicator in verification_message.lower():
                return True
                
        return False
    
    def _detect_clarification(self, verification_message: str) -> bool:
        """Detect if there was a clarification in the exchange."""
        clarification_indicators = [
            "let me clarify", "to clarify", "clarification", "let me be clearer", 
            "to be more specific", "what I meant was", "to elaborate", "let me explain further"
        ]
        
        for indicator in clarification_indicators:
            if indicator in verification_message.lower():
                return True
                
        return False
    
    def extract_content_from_exchange(self, exchange: Tuple[str, str, str]) -> Dict[str, str]:
        """
        Extract the substantive content from a closed-loop exchange.
        
        Args:
            exchange: Tuple of (initial message, acknowledgment, verification)
            
        Returns:
            Dictionary with sender_content and receiver_content keys
        """
        sender_message, receiver_message, verification_message = exchange
        
        # Extract the substantive content
        # For the receiver message, remove the acknowledgment portion
        receiver_content = receiver_message
        acknowledgment_markers = [
            "I received your message", "I acknowledge", "Thank you for your message",
            "I understand", "I've received", "I have received"
        ]
        
        for marker in acknowledgment_markers:
            if marker in receiver_message:
                parts = receiver_message.split(marker, 1)
                if len(parts) > 1:
                    # Find the next paragraph break after the acknowledgment
                    content_parts = parts[1].split("\n\n", 1)
                    if len(content_parts) > 1:
                        receiver_content = content_parts[1]
                    break
        
        # For the verification message, remove the verification portion
        verification_content = verification_message
        verification_markers = [
            "You've understood", "Your understanding is", "You understood", 
            "You've correctly", "Thank you for confirming"
        ]
        
        for marker in verification_markers:
            if marker in verification_message:
                parts = verification_message.split(marker, 1)
                if len(parts) > 1:
                    # Find the next paragraph break after the verification
                    content_parts = parts[1].split("\n\n", 1)
                    if len(content_parts) > 1:
                        verification_content = content_parts[1]
                    break
        
        return {
            "sender_content": sender_message,
            "receiver_content": receiver_content,
            "verification_content": verification_content
        }
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """
        Get metrics on the closed-loop communication effectiveness.
        
        Returns:
            Dictionary with communication metrics
        """
        total_exchanges = len(self.exchanges)
        
        if total_exchanges == 0:
            return {
                "total_exchanges": 0,
                "misunderstanding_rate": 0,
                "clarification_rate": 0,
                "effectiveness_rating": "N/A"
            }
        
        misunderstanding_rate = self.misunderstandings / total_exchanges
        clarification_rate = self.clarifications / total_exchanges
        
        # Determine effectiveness rating
        if misunderstanding_rate < 0.1:
            effectiveness = "high"
        elif misunderstanding_rate < 0.3:
            effectiveness = "medium"
        else:
            effectiveness = "low"
        
        return {
            "total_exchanges": total_exchanges,
            "misunderstandings": self.misunderstandings,
            "clarifications": self.clarifications,
            "misunderstanding_rate": misunderstanding_rate,
            "clarification_rate": clarification_rate,
            "effectiveness_rating": effectiveness
        }