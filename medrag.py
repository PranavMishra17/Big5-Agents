"""
MedRAG implementation adapted for Azure OpenAI integration.
Based on the original MedRAG framework with Azure OpenAI support.
"""

import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union

# Add src path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from utils import RetrievalSystem, DocExtracter
    #from template import *
except ImportError as e:
    logging.warning(f"Could not import MedRAG components: {e}")
    # Create minimal fallbacks
    class RetrievalSystem:
        def __init__(self, *args, **kwargs):
            pass
        def retrieve(self, question, k=32, rrf_k=100):
            return [], []
    
    class DocExtracter:
        def __init__(self, *args, **kwargs):
            pass
        def extract(self, snippets_ids):
            return []

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)

class MedRAG:
    """
    MedRAG class adapted for Azure OpenAI with medical knowledge retrieval.
    """

    def __init__(self, 
                 llm_name: str = "azure_openai",
                 rag: bool = True,
                 follow_up: bool = False,
                 retriever_name: str = "MedCPT",
                 corpus_name: str = "Textbooks",
                 db_dir: str = "./corpus",
                 cache_dir: str = None,
                 corpus_cache: bool = False,
                 HNSW: bool = False,
                 azure_config: Optional[Dict[str, str]] = None):
        """
        Initialize MedRAG with Azure OpenAI configuration.
        """
        self.llm_name = llm_name
        self.rag = rag
        self.follow_up = follow_up
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.azure_config = azure_config or {}
        
        self.logger = logging.getLogger("medrag.core")
        
        # Initialize Azure OpenAI client
        self._init_azure_client()
        
        # Initialize retriever
        self._init_retrieval_system(corpus_cache, HNSW)
        
        # Initialize templates
        self._init_templates()
        
        # Initialize tokenizer and model settings
        self._init_model_settings()
        
        self.logger.info(f"MedRAG initialized: {retriever_name}/{corpus_name}, RAG: {rag}")

    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            if self.llm_name.lower() == "azure_openai":
                # Use Azure OpenAI
                self.client = openai.AzureOpenAI(
                    api_key=self.azure_config.get("api_key"),
                    api_version=self.azure_config.get("api_version", "2024-08-01-preview"),
                    azure_endpoint=self.azure_config.get("azure_endpoint")
                )
                self.model = self.azure_config.get("azure_deployment", "gpt-4")
                self.max_length = 32768 if "gpt-4" in self.model else 16384
                self.context_length = 30000 if "gpt-4" in self.model else 15000
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                
                self.logger.info(f"Azure OpenAI client initialized: {self.model}")
            else:
                raise ValueError(f"Unsupported LLM: {self.llm_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    def _init_retrieval_system(self, corpus_cache: bool, HNSW: bool):
        """Initialize the medical knowledge retrieval system."""
        self.docExt = None
        if self.rag:
            try:
                self.retrieval_system = RetrievalSystem(
                    self.retriever_name, 
                    self.corpus_name, 
                    self.db_dir, 
                    cache=corpus_cache, 
                    HNSW=HNSW
                )
                self.logger.info("RetrievalSystem initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize RetrievalSystem: {str(e)}")
                self.retrieval_system = None
                self.rag = False
                self.logger.warning("Disabling RAG due to retrieval system failure")
        else:
            self.retrieval_system = None

    def _init_templates(self):
        """Initialize prompt templates."""
        try:
            # Import templates from the actual MedRAG
            from utils.prompts import (
                general_cot_system, general_cot,
                general_medrag_system, general_medrag
            )
            self.templates = {
                "cot_system": general_cot_system,
                "cot_prompt": general_cot,
                "medrag_system": general_medrag_system, 
                "medrag_prompt": general_medrag
            }
        except ImportError:
            # Fallback templates
            self.logger.warning("Using fallback templates")
            self.templates = {
                "cot_system": "You are a helpful medical expert. Answer medical questions step-by-step.",
                "cot_prompt": "Question: {question}\nOptions: {options}\nThink step-by-step:",
                "medrag_system": "You are a helpful medical expert. Use the provided documents to answer questions.",
                "medrag_prompt": "Documents: {context}\nQuestion: {question}\nOptions: {options}\nAnswer:"
            }

    def _init_model_settings(self):
        """Initialize model-specific settings."""
        if "gpt-3.5" in self.model or "gpt-35" in self.model:
            self.max_length = 16384
            self.context_length = 15000
        elif "gpt-4" in self.model:
            self.max_length = 32768
            self.context_length = 30000

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Azure OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def answer(self, 
               question: str,
               options: Optional[Dict[str, str]] = None,
               snippets: Optional[List[Dict]] = None,
               snippets_ids: Optional[List[Dict]] = None,
               k: int = 32,
               rrf_k: int = 100,
               save_dir: str = None,
               **kwargs) -> Tuple[Union[str, Dict], List[Dict], List[float]]:
        """
        Generate answer using MedRAG with retrieved medical knowledge.
        """
        try:
            # Format options
            if options is not None:
                if isinstance(options, dict):
                    options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
                else:
                    options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            else:
                options_text = ''

            # Retrieve relevant snippets
            if self.rag:
                if snippets is not None:
                    retrieved_snippets = snippets[:k]
                    scores = [1.0] * len(retrieved_snippets)
                elif snippets_ids is not None:
                    if self.docExt is None:
                        self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                    retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                    scores = [1.0] * len(retrieved_snippets)
                else:
                    if self.retrieval_system is not None:
                        retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
                    else:
                        retrieved_snippets = []
                        scores = []

                # Format contexts
                contexts = []
                if retrieved_snippets:
                    context_parts = []
                    for idx, snippet in enumerate(retrieved_snippets):
                        title = snippet.get("title", f"Document {idx+1}")
                        content = snippet.get("content", snippet.get("contents", ""))
                        context_parts.append(f"Document [{idx+1}] (Title: {title}) {content}")
                    
                    context_text = "\n".join(context_parts)
                    # Truncate to context length
                    if hasattr(self, 'tokenizer'):
                        tokens = self.tokenizer.encode(context_text)
                        if len(tokens) > self.context_length:
                            truncated_tokens = tokens[:self.context_length]
                            context_text = self.tokenizer.decode(truncated_tokens)
                    contexts = [context_text]
                else:
                    contexts = [""]
            else:
                retrieved_snippets = []
                scores = []
                contexts = []

            # Generate answers
            if not self.rag:
                # Chain-of-thought prompting
                if hasattr(self.templates["cot_prompt"], 'render'):
                    prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_text)
                else:
                    prompt_cot = f"Question: {question}\nOptions: {options_text}\nThink step-by-step and provide your answer:"
                
                messages = [
                    {"role": "system", "content": self.templates["cot_system"]},
                    {"role": "user", "content": prompt_cot}
                ]
                ans = self.generate(messages, **kwargs)
            else:
                # RAG prompting
                context = contexts[0] if contexts else ""
                
                if hasattr(self.templates["medrag_prompt"], 'render'):
                    prompt_medrag = self.templates["medrag_prompt"].render(
                        context=context, 
                        question=question, 
                        options=options_text
                    )
                else:
                    prompt_medrag = f"Documents: {context}\nQuestion: {question}\nOptions: {options_text}\nAnswer:"
                
                messages = [
                    {"role": "system", "content": self.templates["medrag_system"]},
                    {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages, **kwargs)

            # Clean and format response
            ans = re.sub(r"\s+", " ", ans.strip())
            
            # Try to parse as JSON
            try:
                parsed_response = json.loads(ans)
                return parsed_response, retrieved_snippets, scores
            except json.JSONDecodeError:
                # Return as structured dict if not valid JSON
                return {
                    "step_by_step_thinking": ans,
                    "answer_choice": self._extract_answer_choice(ans, options)
                }, retrieved_snippets, scores
                
        except Exception as e:
            self.logger.error(f"Error in MedRAG answer generation: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return fallback response
            fallback_answer = {
                "step_by_step_thinking": f"Error occurred during processing: {str(e)}",
                "answer_choice": "A" if options else "Unable to determine"
            }
            return fallback_answer, [], []

    def _extract_answer_choice(self, response_text: str, options: Optional[Dict[str, str]]) -> str:
        """Extract answer choice from response text."""
        if not options:
            return "N/A"
        
        # Look for answer patterns
        import re
        patterns = [
            r'"answer_choice":\s*"([A-D])"',
            r"answer_choice.*?([A-D])",
            r"ANSWER:\s*([A-D])",
            r"answer is\s*([A-D])",
            r"option\s*([A-D])",
            r"\b([A-D])\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                choice = match.group(1).upper()
                if isinstance(options, dict):
                    if choice in options:
                        return choice
                else:
                    if ord(choice) - ord('A') < len(options):
                        return choice
        
        # Default to first option if no clear choice found
        if isinstance(options, dict):
            return list(options.keys())[0] if options else "A"
        else:
            return "A"


class MedRAGRetriever:
    """
    Base class for medical knowledge retrievers.
    This should be implemented with actual retrieval systems.
    """
    
    def __init__(self, corpus_name: str = "Textbooks"):
        self.corpus_name = corpus_name
        self.logger = logging.getLogger("medrag.retriever")
    
    def retrieve(self, query: str, k: int = 16) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve relevant medical knowledge snippets.
        
        Args:
            query: Search query
            k: Number of snippets to retrieve
            
        Returns:
            Tuple of (snippets, relevance_scores)
        """
        raise NotImplementedError("Subclasses must implement retrieve method")


# Example of how to implement actual retrievers
class TextbookRetriever(MedRAGRetriever):
    """
    Retriever for medical textbook content.
    This is a placeholder - implement with actual textbook retrieval.
    """
    
    def __init__(self, corpus_name: str = "Textbooks"):
        super().__init__(corpus_name)
        self.textbook_index = self._load_textbook_index()
    
    def _load_textbook_index(self):
        """Load textbook search index (placeholder)."""
        self.logger.warning("Using placeholder textbook index")
        return {}
    
    def retrieve(self, query: str, k: int = 16) -> Tuple[List[Dict], List[float]]:
        """Retrieve from medical textbooks."""
        # Placeholder implementation
        snippets = []
        scores = []
        
        # In actual implementation, this would:
        # 1. Process the query
        # 2. Search the textbook index
        # 3. Rank results by relevance
        # 4. Return top k results
        
        for i in range(min(k, 3)):  # Return up to 3 placeholder results
            snippet = {
                "id": f"textbook_snippet_{i}",
                "title": f"Medical_Textbook_Chapter_{i}",
                "content": f"Relevant medical information about: {query[:100]}... [Placeholder content]",
                "contents": f"Relevant medical information about: {query[:100]}... [Placeholder content]"
            }
            snippets.append(snippet)
            scores.append(0.8 - (i * 0.1))
        
        return snippets, scores


class PubMedRetriever(MedRAGRetriever):
    """
    Retriever for PubMed medical literature.
    This is a placeholder - implement with actual PubMed API.
    """
    
    def __init__(self, corpus_name: str = "PubMed"):
        super().__init__(corpus_name)
        self.pubmed_api_key = None  # Set if you have PubMed API access
    
    def retrieve(self, query: str, k: int = 16) -> Tuple[List[Dict], List[float]]:
        """Retrieve from PubMed literature."""
        # Placeholder implementation
        snippets = []
        scores = []
        
        # In actual implementation, this would:
        # 1. Query PubMed API
        # 2. Process abstracts and full texts
        # 3. Rank by relevance
        # 4. Return top k results
        
        for i in range(min(k, 2)):  # Return up to 2 placeholder results
            snippet = {
                "id": f"pubmed_{i}",
                "title": f"PubMed_Article_{i}",
                "content": f"Medical research findings related to: {query[:100]}... [Placeholder PubMed content]",
                "contents": f"Medical research findings related to: {query[:100]}... [Placeholder PubMed content]"
            }
            snippets.append(snippet)
            scores.append(0.9 - (i * 0.1))
        
        return snippets, scores


def create_retriever(retriever_name: str, corpus_name: str) -> MedRAGRetriever:
    """
    Factory function to create appropriate retriever.
    
    Args:
        retriever_name: Name of retriever ("MedCPT", "Textbooks", "PubMed")
        corpus_name: Name of corpus to search
        
    Returns:
        Configured retriever instance
    """
    #if retriever_name == "MedCPT":
        # Use mock for now - replace with actual MedCPT implementation
        # return MockMedCPTRetriever(corpus_name)
    if retriever_name == "Textbooks":
        return TextbookRetriever(corpus_name)
    elif retriever_name == "PubMed":
        return PubMedRetriever(corpus_name)
    else:
        raise ValueError(f"Unknown retriever: {retriever_name}")


# Utility functions for MedRAG integration
def format_medical_question(question: str, options: Optional[Dict[str, str]] = None) -> str:
    """Format medical question for retrieval."""
    formatted = question.strip()
    
    if options:
        # Add options context for better retrieval
        options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
        formatted += f" Options: {options_text}"
    
    return formatted


def extract_medical_entities(text: str) -> List[str]:
    """
    Extract medical entities from text for better retrieval.
    This is a placeholder - implement with actual medical NER.
    """
    # Placeholder implementation
    common_medical_terms = [
        "diagnosis", "treatment", "symptoms", "disease", "condition",
        "medication", "therapy", "syndrome", "disorder", "infection"
    ]
    
    entities = []
    text_lower = text.lower()
    
    for term in common_medical_terms:
        if term in text_lower:
            entities.append(term)
    
    return entities


def validate_medical_content(content: str) -> bool:
    """
    Validate that content is medical in nature.
    This is a placeholder - implement with actual medical content validation.
    """
    medical_indicators = [
        "patient", "clinical", "medical", "disease", "treatment",
        "diagnosis", "symptoms", "therapy", "medication", "hospital"
    ]
    
    content_lower = content.lower()
    medical_score = sum(1 for indicator in medical_indicators if indicator in content_lower)
    
    return medical_score >= 2  # At least 2 medical terms


# Example usage and testing
if __name__ == "__main__":
    # Test the MedRAG implementation
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    azure_config = {
        "api_key": "your-azure-api-key",
        "api_version": "2024-08-01-preview",
        "azure_endpoint": "https://your-endpoint.openai.azure.com/",
        "azure_deployment": "gpt-4"
    }
    
    # Example question
    question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
    options = {
        "A": "paralysis of the facial muscles.",
        "B": "paralysis of the facial muscles and loss of taste.",
        "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
        "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
    }
    
    try:
        # Initialize MedRAG
        medrag = MedRAG(
            llm_name="azure_openai",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            azure_config=azure_config
        )
        
        # Get answer
        answer, snippets, scores = medrag.answer(question=question, options=options, k=16)
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Retrieved {len(snippets)} snippets with scores: {scores[:3]}...")
        
    except Exception as e:
        print(f"Error testing MedRAG: {str(e)}")
        print("Note: Ensure Azure OpenAI credentials are properly configured")