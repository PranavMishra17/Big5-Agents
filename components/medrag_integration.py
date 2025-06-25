"""
Fixed MedRAG integration component that properly works with Azure OpenAI.
"""

import logging
import json
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from datetime import datetime

import config


class MedRAGIntegration:
    """
    Fixed MedRAG integration for knowledge enhancement in the agent system.
    """
    
    def __init__(self, 
                 llm_name: str = "azure_openai",
                 retriever_name: str = "MedCPT", 
                 corpus_name: str = "Textbooks",
                 k_retrieval: int = 16,
                 deployment_config: Optional[Dict[str, str]] = None):
        """
        Initialize MedRAG integration.
        """
        self.logger = logging.getLogger("medrag.integration")
        self.llm_name = llm_name
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.k_retrieval = k_retrieval
        self.deployment_config = deployment_config or self._get_default_deployment()
        
        # Thread-safe storage for retrieved knowledge
        self._knowledge_cache = {}
        self._cache_lock = threading.Lock()
        
        # Initialize MedRAG with error handling
        self.medrag = None
        self._initialization_error = None
        self._initialize_medrag()
        
        self.logger.info(f"MedRAG integration initialized with {retriever_name}/{corpus_name}")
    
    def _get_default_deployment(self) -> Dict[str, str]:
        """Get default deployment configuration."""
        deployments = config.get_all_deployments()
        return deployments[0] if deployments else {
            "name": "default",
            "deployment": "gpt-4",
            "api_key": config.AZURE_API_KEY,
            "endpoint": config.AZURE_ENDPOINT,
            "api_version": "2024-08-01-preview"
        }
    
    def _initialize_medrag(self):
        """Initialize MedRAG with proper error handling."""
        try:
            # Import our fixed MedRAG implementation
            from medrag import MedRAG
            
            # Configure for Azure OpenAI using the deployment config
            azure_config = {
                "api_key": self.deployment_config["api_key"],
                "api_version": self.deployment_config["api_version"],
                "azure_endpoint": self.deployment_config["endpoint"],
                "azure_deployment": self.deployment_config["deployment"]
            }
            
            # Initialize MedRAG with Azure OpenAI configuration
            self.medrag = MedRAG(
                llm_name="azure_openai",
                rag=True,
                retriever_name=self.retriever_name,
                corpus_name=self.corpus_name,
                azure_config=azure_config,
                corpus_cache=True  # Enable caching for performance
            )
            
            self.logger.info("MedRAG successfully initialized with Azure OpenAI")
            
        except ImportError as e:
            self._initialization_error = f"MedRAG not available: {str(e)}"
            self.logger.error(f"Failed to import MedRAG: {str(e)}")
            
        except Exception as e:
            self._initialization_error = f"MedRAG initialization failed: {str(e)}"
            self.logger.error(f"Failed to initialize MedRAG: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def is_available(self) -> bool:
        """Check if MedRAG is available and properly initialized."""
        return self.medrag is not None and self._initialization_error is None
    
    def get_initialization_error(self) -> Optional[str]:
        """Get initialization error if any."""
        return self._initialization_error
    


    def retrieve_knowledge(self, 
                        question: str, 
                        options: List[str] = None,
                        question_id: str = None) -> Dict[str, Any]:
        """
        Retrieve relevant medical knowledge for a question using MedRAG.
        IMPROVED VERSION with better medical question handling.
        """
        if not self.is_available():
            self.logger.warning(f"MedRAG not available: {self._initialization_error}")
            return {
                "available": False,
                "error": self._initialization_error,
                "knowledge_snippets": [],
                "summary": "MedRAG retrieval not available"
            }
        
        # Check cache first
        cache_key = self._generate_cache_key(question, options)
        with self._cache_lock:
            if cache_key in self._knowledge_cache:
                self.logger.debug(f"Retrieved knowledge from cache for question: {question[:50]}...")
                cached_result = self._knowledge_cache[cache_key].copy()
                cached_result["cached"] = True
                return cached_result
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Retrieving knowledge for question: {question[:100]}...")
            
            # IMPROVEMENT: Better query preprocessing for medical questions
            enhanced_query = self._enhance_medical_query(question, options)
            
            # Format options for MedRAG
            formatted_options = self._format_options(options) if options else None
            
            # Call MedRAG with enhanced query
            answer, snippets, scores = self._call_medrag_with_timeout(
                question=enhanced_query,
                options=formatted_options,
                k=min(self.k_retrieval, 10),  # Reduce to 10 for better quality
                timeout=45  # Increase timeout for better results
            )
            
            # IMPROVEMENT: Filter and rank snippets by relevance
            filtered_snippets, filtered_scores = self._filter_medical_snippets(
                snippets, scores, question, options
            )
            
            # Process and structure the retrieved knowledge
            knowledge_result = self._process_retrieved_knowledge(
                question=question,
                answer=answer,
                snippets=filtered_snippets,
                scores=filtered_scores
            )
            
            # Cache the result
            with self._cache_lock:
                self._knowledge_cache[cache_key] = knowledge_result.copy()
            
            retrieval_time = time.time() - start_time
            knowledge_result["retrieval_time"] = retrieval_time
            knowledge_result["cached"] = False
            
            self.logger.info(f"Successfully retrieved {len(filtered_snippets)} knowledge snippets in {retrieval_time:.2f}s")
            
            return knowledge_result
            
        except Exception as e:
            error_msg = f"Failed to retrieve knowledge: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return {
                "available": False,
                "error": error_msg,
                "knowledge_snippets": [],
                "summary": f"Knowledge retrieval failed: {str(e)}",
                "retrieval_time": time.time() - start_time
            }


    def _enhance_medical_query(self, question: str, options: List[str] = None) -> str:
        """Enhance the query for better medical knowledge retrieval."""
        enhanced_query = question
        
        # Extract key medical terms
        medical_terms = self._extract_medical_terms(question)
        
        # Add option context if available
        if options:
            option_terms = []
            for option in options:
                # Extract medical terms from options
                opt_terms = self._extract_medical_terms(option)
                option_terms.extend(opt_terms)
            
            if option_terms:
                unique_terms = list(set(option_terms))
                enhanced_query += f" {' '.join(unique_terms[:5])}"  # Add top 5 terms
        
        return enhanced_query

    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract key medical terms from text."""
        # Common medical term patterns
        medical_patterns = [
            r'\b\w*itis\b',      # inflammation terms
            r'\b\w*osis\b',      # condition terms  
            r'\b\w*emia\b',      # blood terms
            r'\b\w*pathy\b',     # disease terms
            r'\b\w*therapy\b',   # treatment terms
            r'\b\w*gram\b',      # imaging terms
            r'\b\w*scopy\b',     # procedure terms
        ]
        
        terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        # Add specific medical keywords
        medical_keywords = [
            'antibodies', 'encephalitis', 'temporal', 'lobe', 'mri', 'csf', 
            'eeg', 'seizure', 'memory', 'confusion', 'behavioral', 'lymphocytic',
            'pleocytosis', 'hyperintensities', 'nmda', 'lgi1', 'gaba', 'ampa'
        ]
        
        text_lower = text.lower()
        for keyword in medical_keywords:
            if keyword in text_lower:
                terms.append(keyword)
        
        return list(set(terms))

    def _filter_medical_snippets(self, snippets: List[Dict], scores: List[float], 
                            question: str, options: List[str] = None) -> Tuple[List[Dict], List[float]]:
        """Filter and improve ranking of medical snippets."""
        if not snippets:
            return [], []
        
        question_lower = question.lower()
        filtered_pairs = []
        
        for snippet, score in zip(snippets, scores):
            content = snippet.get("content", "").lower()
            title = snippet.get("title", "").lower()
            
            # Calculate enhanced relevance score
            enhanced_score = score
            
            # Boost for title matches
            title_boost = sum(1 for term in question_lower.split() 
                            if len(term) > 3 and term in title) * 0.1
            enhanced_score += title_boost
            
            # Boost for medical term matches
            medical_terms = self._extract_medical_terms(question)
            term_boost = sum(1 for term in medical_terms 
                            if term.lower() in content) * 0.15
            enhanced_score += term_boost
            
            # Boost for option-related content
            if options:
                option_boost = 0
                for option in options:
                    option_terms = option.lower().split()
                    option_boost += sum(1 for term in option_terms 
                                    if len(term) > 3 and term in content) * 0.05
                enhanced_score += option_boost
            
            # Filter out very low relevance snippets
            if enhanced_score > 0.1:
                filtered_pairs.append((snippet, enhanced_score))
        
        # Sort by enhanced score and return top results
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 snippets
        top_pairs = filtered_pairs[:5]
        filtered_snippets = [pair[0] for pair in top_pairs]
        filtered_scores = [pair[1] for pair in top_pairs]
        
        return filtered_snippets, filtered_scores

    def _format_options(self, options: List[str]) -> Dict[str, str]:
        """Format options for MedRAG input."""
        if not options:
            return {}
        
        formatted = {}
        for i, option in enumerate(options):
            # Handle options that may already have labels (A. text) or just text
            if '. ' in option and len(option.split('. ', 1)[0]) == 1:
                # Already formatted as "A. text"
                label = option.split('. ', 1)[0]
                text = option.split('. ', 1)[1]
            else:
                # Just text, add label
                label = chr(65 + i)  # A, B, C, D, ...
                text = option
            
            formatted[label] = text
        
        return formatted
    
    def _call_medrag_with_timeout(self, question: str, options: Dict[str, str], k: int, timeout: int) -> Tuple[Any, List, List]:
        """Call MedRAG with timeout handling."""
        result = [None, [], []]
        exception = [None]
        
        def target():
            try:
                answer, snippets, scores = self.medrag.answer(
                    question=question,
                    options=options,
                    k=k
                )
                result[0] = answer
                result[1] = snippets
                result[2] = scores
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            self.logger.warning(f"MedRAG call timed out after {timeout} seconds")
            raise TimeoutError(f"MedRAG retrieval timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0], result[1], result[2]
    
    def _process_retrieved_knowledge(self, 
                                   question: str,
                                   answer: Any,
                                   snippets: List[Dict],
                                   scores: List[float]) -> Dict[str, Any]:
        """Process and structure retrieved knowledge."""
        # Extract key information from snippets
        knowledge_snippets = []
        for i, (snippet, score) in enumerate(zip(snippets, scores)):
            processed_snippet = {
                "id": snippet.get("id", f"snippet_{i}"),
                "title": snippet.get("title", "Unknown Source"),
                "content": snippet.get("content", snippet.get("contents", "")),
                "relevance_score": float(score) if score is not None else 0.0,
                "rank": i + 1
            }
            knowledge_snippets.append(processed_snippet)
        
        # Create summary of retrieved knowledge
        summary = self._create_knowledge_summary(knowledge_snippets, question)
        
        # Extract MedRAG answer insights if available
        medrag_insights = self._extract_medrag_insights(answer)
        
        return {
            "available": True,
            "success": True,
            "question": question,
            "knowledge_snippets": knowledge_snippets,
            "summary": summary,
            "medrag_insights": medrag_insights,
            "retrieval_metadata": {
                "num_snippets": len(knowledge_snippets),
                "retriever": self.retriever_name,
                "corpus": self.corpus_name,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_knowledge_summary(self, snippets: List[Dict], question: str) -> str:
        """Create a concise summary of retrieved knowledge."""
        if not snippets:
            return "No relevant knowledge retrieved."
        
        # Get top 3 most relevant snippets for summary
        top_snippets = sorted(snippets, key=lambda x: x["relevance_score"], reverse=True)[:3]
        
        summary_parts = [
            f"Retrieved {len(snippets)} relevant medical knowledge snippets.",
            f"Top sources: {', '.join([s['title'] for s in top_snippets])}",
        ]
        
        # Add brief content summary
        if top_snippets:
            top_content = top_snippets[0]["content"][:200] + "..." if len(top_snippets[0]["content"]) > 200 else top_snippets[0]["content"]
            summary_parts.append(f"Key information: {top_content}")
        
        return " ".join(summary_parts)
    
    def _extract_medrag_insights(self, answer: Any) -> Dict[str, Any]:
        """Extract insights from MedRAG answer."""
        if not answer:
            return {}
        
        try:
            if isinstance(answer, str):
                # Try to parse as JSON
                try:
                    parsed_answer = json.loads(answer)
                    return {
                        "reasoning": parsed_answer.get("step_by_step_thinking", ""),
                        "suggested_answer": parsed_answer.get("answer_choice", ""),
                        "confidence": parsed_answer.get("confidence", None)
                    }
                except json.JSONDecodeError:
                    return {"reasoning": answer}
            
            elif isinstance(answer, dict):
                return {
                    "reasoning": answer.get("step_by_step_thinking", ""),
                    "suggested_answer": answer.get("answer_choice", ""),
                    "confidence": answer.get("confidence", None)
                }
            
            else:
                return {"raw_answer": str(answer)}
                
        except Exception as e:
            self.logger.warning(f"Failed to extract MedRAG insights: {str(e)}")
            return {"error": f"Failed to parse MedRAG answer: {str(e)}"}
    
    def _generate_cache_key(self, question: str, options: List[str] = None) -> str:
        """Generate cache key for question and options."""
        import hashlib
        
        content = question
        if options:
            content += "|" + "|".join(options)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def enhance_agent_knowledge(self, 
                               agent,
                               retrieved_knowledge: Dict[str, Any]) -> bool:
        """
        Enhance individual agent's knowledge base with retrieved knowledge.
        """
        if not retrieved_knowledge.get("available", False):
            self.logger.warning("Cannot enhance agent knowledge: no valid knowledge retrieved")
            return False
        
        try:
            # Add knowledge to agent's knowledge base
            agent.add_to_knowledge_base("medrag_knowledge", {
                "summary": retrieved_knowledge["summary"],
                "snippets": retrieved_knowledge["knowledge_snippets"][:5],  # Top 5 snippets
                "insights": retrieved_knowledge.get("medrag_insights", {}),
                "retrieval_metadata": retrieved_knowledge.get("retrieval_metadata", {})
            })
            
            # Add specific medical context if available
            if retrieved_knowledge["knowledge_snippets"]:
                medical_context = self._create_medical_context(retrieved_knowledge["knowledge_snippets"])
                agent.add_to_knowledge_base("medical_context", medical_context)
            
            self.logger.debug(f"Enhanced {agent.role} knowledge with MedRAG retrieval")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enhance agent knowledge: {str(e)}")
            return False
    
    def enhance_shared_mental_model(self, 
                                  mental_model,
                                  retrieved_knowledge: Dict[str, Any]) -> bool:
        """
        Enhance shared mental model with retrieved knowledge.
        """
        if not retrieved_knowledge.get("available", False):
            self.logger.warning("Cannot enhance shared mental model: no valid knowledge retrieved")
            return False
        
        try:
            # Add knowledge to shared mental model
            mental_model.shared_understanding["medrag_knowledge"] = {
                "summary": retrieved_knowledge["summary"],
                "key_snippets": retrieved_knowledge["knowledge_snippets"][:3],  # Top 3 for sharing
                "medical_insights": retrieved_knowledge.get("medrag_insights", {})
            }
            
            self.logger.debug("Enhanced shared mental model with MedRAG knowledge")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enhance shared mental model: {str(e)}")
            return False
    
    def _create_medical_context(self, snippets: List[Dict]) -> Dict[str, Any]:
        """Create structured medical context from snippets."""
        if not snippets:
            return {}
        
        # Group snippets by source/title
        sources = {}
        for snippet in snippets[:5]:  # Top 5 snippets
            title = snippet.get("title", "Unknown")
            if title not in sources:
                sources[title] = []
            sources[title].append({
                "content": snippet["content"][:300] + "..." if len(snippet["content"]) > 300 else snippet["content"],
                "relevance": snippet.get("relevance_score", 0.0)
            })
        
        return {
            "sources": sources,
            "num_references": len(snippets),
            "highest_relevance": max([s.get("relevance_score", 0.0) for s in snippets])
        }
    
    def get_enhancement_prompt_addition(self, retrieved_knowledge: Dict[str, Any]) -> str:
        """
        Generate prompt addition for agents with retrieved knowledge.
        """
        if not retrieved_knowledge.get("available", False):
            return ""
        
        snippets = retrieved_knowledge.get("knowledge_snippets", [])
        if not snippets:
            return ""
        
        # Format prompt addition
        prompt_addition = "\n\n=== RETRIEVED MEDICAL KNOWLEDGE ===\n"
        prompt_addition += f"The following medical knowledge has been retrieved to assist with this question:\n\n"
        
        # Add top 3 most relevant snippets
        top_snippets = sorted(snippets, key=lambda x: x.get("relevance_score", 0), reverse=True)[:3]
        
        for i, snippet in enumerate(top_snippets, 1):
            content = snippet["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            
            prompt_addition += f"{i}. {snippet.get('title', 'Medical Reference')}: {content}\n\n"
        
        # Add MedRAG insights if available
        insights = retrieved_knowledge.get("medrag_insights", {})
        if insights.get("reasoning"):
            prompt_addition += f"Additional clinical reasoning: {insights['reasoning'][:200]}...\n\n"
        
        prompt_addition += "Consider this retrieved medical knowledge in your analysis and decision-making.\n"
        prompt_addition += "=== END RETRIEVED KNOWLEDGE ===\n"
        
        return prompt_addition
    
    def clear_cache(self):
        """Clear the knowledge cache."""
        with self._cache_lock:
            self._knowledge_cache.clear()
        self.logger.info("Cleared MedRAG knowledge cache")


def create_medrag_integration(deployment_config: Optional[Dict[str, str]] = None,
                            retriever_name: str = "MedCPT",
                            corpus_name: str = "Textbooks") -> Optional[MedRAGIntegration]:
    """
    Factory function to create MedRAG integration with error handling.
    """
    try:
        integration = MedRAGIntegration(
            deployment_config=deployment_config,
            retriever_name=retriever_name,
            corpus_name=corpus_name
        )
        
        if integration.is_available():
            return integration
        else:
            logging.getLogger("medrag").warning(f"MedRAG integration not available: {integration.get_initialization_error()}")
            return None
            
    except Exception as e:
        logging.getLogger("medrag").error(f"Failed to create MedRAG integration: {str(e)}")
        return None