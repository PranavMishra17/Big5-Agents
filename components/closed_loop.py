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
        Facilitate a complete closed-loop communication exchange.
        
        Args:
            sender: The agent sending the initial message
            receiver: The agent receiving the message
            initial_message: The content of the initial message
            
        Returns:
            Tuple containing (initial message, acknowledgment, verification)
        """
        self.logger.info(f"Starting closed-loop exchange: {sender.role} -> {receiver.role}")
        
        # Step 1: Sender initiates message
        sender_message = sender.chat(initial_message)
        self.logger.info(f"Step 1 - Sender message sent: {sender_message[:50]}...")
        
        # Step 2: Receiver acknowledges and confirms understanding
        receiver_prompt = COMMUNICATION_PROMPTS["receiver_acknowledgment"].format(
            sender_role=sender.role,
            sender_message=sender_message
        )
    

        receiver_message = receiver.chat(receiver_prompt)
        self.logger.info(f"Step 2 - Receiver acknowledgment: {receiver_message[:50]}...")
        
        # Step 3: Sender verifies message was received correctly
        verification_prompt = COMMUNICATION_PROMPTS["sender_verification"].format(
            receiver_role=receiver.role,
            sent_message=sender_message,
            response_message=receiver_message
        ) 
        verification_message = sender.chat(verification_prompt)
        self.logger.info(f"Step 3 - Sender verification: {verification_message[:50]}...")
        
        # Track this exchange for metrics
        exchange_data = {
            "sender_role": sender.role,
            "receiver_role": receiver.role,
            "initial_message": sender_message,
            "acknowledgment": receiver_message,
            "verification": verification_message,
            "has_misunderstanding": self._detect_misunderstanding(verification_message),
            "has_clarification": self._detect_clarification(verification_message)
        }
        
        self.exchanges.append(exchange_data)
        
        # Update metrics
        if exchange_data["has_misunderstanding"]:
            self.misunderstandings += 1
        if exchange_data["has_clarification"]:
            self.clarifications += 1
        
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