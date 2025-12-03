"""
Self-Supervised Prompt Optimization (SPO) Framework
Based on MetaGPT implementation for automated prompt engineering
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from llm_client import create_llm_client
import os

@dataclass
class PromptOptimizationConfig:
    """Configuration for prompt optimization"""
    max_iterations: int = 5
    optimization_model: Optional[str] = None  # Read from OPTIMIZATION_MODEL env var
    execution_model: Optional[str] = None  # Read from EXECUTION_MODEL env var
    evaluation_model: Optional[str] = None  # Read from EVALUATION_MODEL env var
    temperature: float = 0.7
    max_tokens: int = 0
    timeout: int = 30
    
    def __post_init__(self):
        """Set default models from environment variables if not provided"""
        if self.optimization_model is None:
            self.optimization_model = os.getenv("OPTIMIZATION_MODEL", "openai/gpt-4o")
        if self.execution_model is None:
            self.execution_model = os.getenv("EXECUTION_MODEL", "openai/gpt-4o-mini")
        if self.evaluation_model is None:
            self.evaluation_model = os.getenv("EVALUATION_MODEL", "openai/gpt-4o")

@dataclass
class OptimizationResult:
    """Result of a single optimization iteration"""
    iteration: int
    original_prompt: str
    optimized_prompt: str
    performance_score: float
    feedback: str
    execution_time: float

class SPOFramework:
    """Self-Supervised Prompt Optimization Framework"""
    
    def __init__(self, config: PromptOptimizationConfig, api_key: Optional[str] = None):
        self.config = config
        
        # Create OpenRouter LLM client
        self.client = create_llm_client(api_key=api_key)
        
        self.logger = self._setup_logging()
        self.optimization_history: List[OptimizationResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("SPO")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    async def optimize_prompt(
        self, 
        initial_prompt: str, 
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Main optimization pipeline
        
        Args:
            initial_prompt: The prompt to optimize
            task_description: Description of what the prompt should accomplish
            sample_inputs: Sample inputs for testing
            expected_outputs: Expected outputs for evaluation (optional)
            
        Returns:
            OptimizationResult with the best performing prompt
        """
        self.logger.info("Starting prompt optimization process")
        self.logger.info(f"Task description: {task_description}")
        current_prompt = initial_prompt
        best_result = None
        
        for iteration in range(self.config.max_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}/{self.config.max_iterations}")
            
            start_time = time.time()
            
            # Generate optimized prompt
            optimized_prompt = await self._generate_optimized_prompt(
                current_prompt, task_description, iteration
            )
            
            # Evaluate performance
            score, feedback = await self._evaluate_prompt_performance(
                optimized_prompt, sample_inputs, expected_outputs, task_description
            )
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                iteration=iteration + 1,
                original_prompt=current_prompt,
                optimized_prompt=optimized_prompt,
                performance_score=score,
                feedback=feedback,
                execution_time=execution_time
            )
            
            self.optimization_history.append(result)
            
            # Update best result if this iteration performed better
            if best_result is None or score > best_result.performance_score:
                best_result = result
                current_prompt = optimized_prompt
                
            self.logger.info(f"Iteration {iteration + 1} score: {score:.2f}")
            
            # Early stopping if performance is excellent
            if score >= 0.95:
                self.logger.info("Early stopping: Excellent performance achieved")
                break
                
        self.logger.info(f"Optimization complete. Best score: {best_result.performance_score:.2f}")
        return best_result
    
    async def _generate_optimized_prompt(
        self, 
        current_prompt: str, 
        task_description: str, 
        iteration: int
    ) -> str:
        """Generate an optimized version of the current prompt"""
        
        optimization_history_context = ""
        if self.optimization_history:
            recent_history = self.optimization_history[-3:]  # Last 3 iterations
            optimization_history_context = "\nPrevious optimization attempts:\n"
            for hist in recent_history:
                optimization_history_context += f"- Iteration {hist.iteration}: Score {hist.performance_score:.2f}\n"
                optimization_history_context += f"  Feedback: {hist.feedback}\n"
        
        system_prompt = f"""You are an expert prompt engineer specializing in optimizing prompts for better performance.

Your task is to improve the given prompt to better accomplish: {task_description}

Guidelines for optimization:
1. Make the prompt more specific and clear
2. Add relevant examples if beneficial
3. Specify desired output format clearly
4. Include handling for edge cases
5. Use clear, actionable language
6. Maintain the core intent while improving clarity and effectiveness

Current iteration: {iteration + 1}
{optimization_history_context}

Analyze the current prompt and provide an improved version that addresses any shortcomings while maintaining the original intent."""

        user_prompt = f"""Current prompt to optimize:
---
{current_prompt}
---

Please provide an optimized version of this prompt that will perform better for the task: {task_description}

Return only the optimized prompt without additional explanation."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.optimization_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            optimized_prompt = response.choices[0].message.content.strip()
            return optimized_prompt
            
        except Exception as e:
            self.logger.error(f"Error generating optimized prompt: {str(e)}")
            return current_prompt
    
    async def _evaluate_prompt_performance(
        self,
        prompt: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]],
        task_description: str
    ) -> Tuple[float, str]:
        """Evaluate the performance of a prompt"""
        
        total_score = 0.0
        evaluation_details = []
        
        for i, sample_input in enumerate(sample_inputs):
            # Execute the prompt with sample input
            try:
                execution_result = await self._execute_prompt(prompt, sample_input)
                
                # Evaluate the result
                if expected_outputs and i < len(expected_outputs):
                    score = await self._score_against_expected(
                        execution_result, expected_outputs[i], task_description
                    )
                else:
                    score = await self._score_general_quality(
                        execution_result, sample_input, task_description
                    )
                
                total_score += score
                evaluation_details.append(f"Input {i+1}: {score:.2f}")
                
            except Exception as e:
                self.logger.warning(f"Error executing prompt for input {i+1}: {str(e)}")
                evaluation_details.append(f"Input {i+1}: 0.0 (execution error)")
        
        average_score = total_score / len(sample_inputs) if sample_inputs else 0.0
        feedback = f"Average score: {average_score:.2f}. Details: {'; '.join(evaluation_details)}"
        
        return average_score, feedback
    
    async def _execute_prompt(self, prompt: str, input_text: str) -> str:
        """Execute a prompt with given input"""
        try:
            response = self.client.chat_completions_create(
                model=self.config.execution_model,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nInput: {input_text}"}
                ],
                temperature=0.1,  # Lower temperature for more consistent execution
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error executing prompt: {str(e)}")
            raise
    
    async def _score_against_expected(
        self, 
        actual_output: str, 
        expected_output: str, 
        task_description: str
    ) -> float:
        """Score output against expected result"""
        
        evaluation_prompt = f"""You are evaluating the quality of an AI system's output for the task: {task_description}

Expected output:
---
{expected_output}
---

Actual output:
---
{actual_output}
---

Please score the actual output on a scale of 0.0 to 1.0 based on:
1. Accuracy compared to expected output
2. Completeness
3. Relevance to the task
4. Format compliance

Return only a number between 0.0 and 1.0 representing the score."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.evaluation_model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            score_text = response.choices[0].message.content.strip()
            print(f"Score text: {score_text}")
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.warning(f"Error scoring output: {str(e)}", exc_info=True)
            return 0.5  # Default middle score on error
    
    async def _score_general_quality(
        self, 
        output: str, 
        input_text: str, 
        task_description: str
    ) -> float:
        """Score output quality when no expected output is available"""
        
        evaluation_prompt = f"""You are evaluating the quality of an AI system's output for the task: {task_description}

Input provided:
---
{input_text}
---

Output generated:
---
{output}
---

Please score the output on a scale of 0.0 to 1.0 based on:
1. Relevance to the input and task
2. Completeness and informativeness
3. Clarity and coherence
4. Accuracy (if verifiable)
5. Format appropriateness

Return only a number between 0.0 and 1.0 representing the score."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.evaluation_model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error scoring output quality: {str(e)}")
            return 0.5
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process"""
        if not self.optimization_history:
            return {"status": "No optimization performed"}
        
        best_result = max(self.optimization_history, key=lambda x: x.performance_score)
        
        return {
            "total_iterations": len(self.optimization_history),
            "best_score": best_result.performance_score,
            "best_iteration": best_result.iteration,
            "improvement": best_result.performance_score - self.optimization_history[0].performance_score,
            "total_time": sum(r.execution_time for r in self.optimization_history),
            "best_prompt": best_result.optimized_prompt
        }
    
    def export_results(self, filename: str):
        """Export optimization results to JSON or YAML file"""
        from yaml_parser import YAMLParser
        
        results_data = {
            "config": asdict(self.config),
            "optimization_history": [asdict(result) for result in self.optimization_history],
            "summary": self.get_optimization_summary()
        }
        
        if YAMLParser.should_save_as_yaml(filename):
            YAMLParser.save_results_yaml(results_data, filename)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results exported to {filename}")