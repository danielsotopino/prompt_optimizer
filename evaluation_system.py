"""
Advanced Evaluation System for Prompt Optimization
Implements comprehensive evaluation metrics and scoring mechanisms
"""
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI
import time
import numpy as np
from abc import ABC, abstractmethod

class EvaluationMetric(Enum):
    """Available evaluation metrics"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance" 
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    SAFETY = "safety"
    ADHERENCE_TO_FORMAT = "adherence_to_format"
    ROBUSTNESS = "robustness"

@dataclass
class EvaluationCriteria:
    """Evaluation criteria configuration"""
    metric: EvaluationMetric
    weight: float
    description: str
    scoring_function: Optional[str] = None

@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    metric: EvaluationMetric
    score: float
    explanation: str
    confidence: float
    evaluation_time: float

@dataclass
class ComprehensiveEvaluation:
    """Complete evaluation result with all metrics"""
    overall_score: float
    metric_scores: Dict[EvaluationMetric, EvaluationResult]
    weighted_score: float
    evaluation_summary: str
    total_evaluation_time: float

class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    @abstractmethod
    async def evaluate(self, prompt: str, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationResult:
        """Evaluate output based on specific metric"""
        pass

class AccuracyEvaluator(BaseEvaluator):
    """Evaluates accuracy of the output"""
    
    async def evaluate(self, prompt: str, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationResult:
        start_time = time.time()
        
        if expected:
            # Compare against expected output
            evaluation_prompt = f"""Evaluate the accuracy of the AI output compared to the expected result.

Expected Output:
{expected}

Actual Output:
{output}

Rate the accuracy on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect match or equivalent meaning
- 0.8-0.9 = Very close, minor differences
- 0.6-0.7 = Generally correct with some errors
- 0.4-0.5 = Partially correct
- 0.0-0.3 = Mostly incorrect

Return your response in this format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Explanation: [detailed explanation]"""
        else:
            # Evaluate factual accuracy without expected output
            evaluation_prompt = f"""Evaluate the factual accuracy and correctness of this AI output.

Input: {input_text}
Output: {output}

Consider:
- Factual correctness
- Logical consistency
- Absence of contradictions
- Appropriate use of information

Rate the accuracy on a scale of 0.0 to 1.0.

Return your response in this format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Explanation: [detailed explanation]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            score, confidence, explanation = self._parse_evaluation_response(result_text)
            
            return EvaluationResult(
                metric=EvaluationMetric.ACCURACY,
                score=score,
                explanation=explanation,
                confidence=confidence,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            return EvaluationResult(
                metric=EvaluationMetric.ACCURACY,
                score=0.5,
                explanation=f"Evaluation error: {str(e)}",
                confidence=0.0,
                evaluation_time=time.time() - start_time
            )
    
    def _parse_evaluation_response(self, response: str) -> Tuple[float, float, str]:
        """Parse the structured evaluation response"""
        try:
            score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', response)
            confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', response)
            explanation_match = re.search(r'Explanation:\s*(.+)', response, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            return score, confidence, explanation
            
        except Exception:
            return 0.5, 0.0, "Failed to parse evaluation response"

class RelevanceEvaluator(BaseEvaluator):
    """Evaluates relevance of the output to the input and task"""
    
    async def evaluate(self, prompt: str, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationResult:
        start_time = time.time()
        
        evaluation_prompt = f"""Evaluate how relevant the AI output is to the given input and task.

Task/Prompt: {prompt}
Input: {input_text}
Output: {output}

Consider:
- Direct relevance to the input
- Addressing the task requirements
- Staying on topic
- Appropriate level of detail

Rate the relevance on a scale of 0.0 to 1.0.

Return your response in this format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Explanation: [detailed explanation]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            score, confidence, explanation = AccuracyEvaluator(self.client)._parse_evaluation_response(result_text)
            
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE,
                score=score,
                explanation=explanation,
                confidence=confidence,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE,
                score=0.5,
                explanation=f"Evaluation error: {str(e)}",
                confidence=0.0,
                evaluation_time=time.time() - start_time
            )

class ClarityEvaluator(BaseEvaluator):
    """Evaluates clarity and readability of the output"""
    
    async def evaluate(self, prompt: str, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationResult:
        start_time = time.time()
        
        evaluation_prompt = f"""Evaluate the clarity and readability of this AI output.

Output: {output}

Consider:
- Clear and understandable language
- Logical structure and organization
- Appropriate level of detail
- Absence of ambiguity
- Professional tone

Rate the clarity on a scale of 0.0 to 1.0.

Return your response in this format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Explanation: [detailed explanation]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            score, confidence, explanation = AccuracyEvaluator(self.client)._parse_evaluation_response(result_text)
            
            return EvaluationResult(
                metric=EvaluationMetric.CLARITY,
                score=score,
                explanation=explanation,
                confidence=confidence,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            return EvaluationResult(
                metric=EvaluationMetric.CLARITY,
                score=0.5,
                explanation=f"Evaluation error: {str(e)}",
                confidence=0.0,
                evaluation_time=time.time() - start_time
            )

class FormatAdherenceEvaluator(BaseEvaluator):
    """Evaluates adherence to specified output format"""
    
    def __init__(self, client: OpenAI, expected_format: str, model: str = "gpt-4o"):
        super().__init__(client, model)
        self.expected_format = expected_format
    
    async def evaluate(self, prompt: str, input_text: str, output: str, expected: Optional[str] = None) -> EvaluationResult:
        start_time = time.time()
        
        evaluation_prompt = f"""Evaluate how well the AI output follows the specified format.

Expected Format: {self.expected_format}
Actual Output: {output}

Consider:
- Exact format compliance
- Required elements present
- Correct structure
- Proper formatting

Rate the format adherence on a scale of 0.0 to 1.0.

Return your response in this format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Explanation: [detailed explanation]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            score, confidence, explanation = AccuracyEvaluator(self.client)._parse_evaluation_response(result_text)
            
            return EvaluationResult(
                metric=EvaluationMetric.ADHERENCE_TO_FORMAT,
                score=score,
                explanation=explanation,
                confidence=confidence,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            return EvaluationResult(
                metric=EvaluationMetric.ADHERENCE_TO_FORMAT,
                score=0.5,
                explanation=f"Evaluation error: {str(e)}",
                confidence=0.0,
                evaluation_time=time.time() - start_time
            )

class ComprehensiveEvaluationSystem:
    """System for comprehensive evaluation using multiple metrics"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.evaluators = {}
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize all available evaluators"""
        self.evaluators = {
            EvaluationMetric.ACCURACY: AccuracyEvaluator(self.client, self.model),
            EvaluationMetric.RELEVANCE: RelevanceEvaluator(self.client, self.model),
            EvaluationMetric.CLARITY: ClarityEvaluator(self.client, self.model)
        }
    
    def add_format_evaluator(self, expected_format: str):
        """Add a format adherence evaluator"""
        self.evaluators[EvaluationMetric.ADHERENCE_TO_FORMAT] = FormatAdherenceEvaluator(
            self.client, expected_format, self.model
        )
    
    async def comprehensive_evaluate(
        self,
        prompt: str,
        input_text: str,
        output: str,
        expected: Optional[str] = None,
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None
    ) -> ComprehensiveEvaluation:
        """Perform comprehensive evaluation using all configured metrics"""
        
        start_time = time.time()
        
        # Use default criteria if none provided
        if evaluation_criteria is None:
            evaluation_criteria = [
                EvaluationCriteria(EvaluationMetric.ACCURACY, 0.4, "Correctness of the output"),
                EvaluationCriteria(EvaluationMetric.RELEVANCE, 0.3, "Relevance to input and task"),
                EvaluationCriteria(EvaluationMetric.CLARITY, 0.3, "Clarity and readability")
            ]
        
        # Run evaluations for each criterion
        evaluation_tasks = []
        for criteria in evaluation_criteria:
            if criteria.metric in self.evaluators:
                task = self.evaluators[criteria.metric].evaluate(prompt, input_text, output, expected)
                evaluation_tasks.append(task)
        
        results = await asyncio.gather(*evaluation_tasks)
        
        # Organize results by metric
        metric_scores = {}
        for result in results:
            metric_scores[result.metric] = result
        
        # Calculate weighted overall score
        weighted_score = 0.0
        total_weight = 0.0
        
        for criteria in evaluation_criteria:
            if criteria.metric in metric_scores:
                weighted_score += metric_scores[criteria.metric].score * criteria.weight
                total_weight += criteria.weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Calculate simple average score
        overall_score = np.mean([result.score for result in results]) if results else 0.0
        
        # Generate evaluation summary
        summary = self._generate_evaluation_summary(metric_scores, evaluation_criteria)
        
        return ComprehensiveEvaluation(
            overall_score=overall_score,
            metric_scores=metric_scores,
            weighted_score=weighted_score,
            evaluation_summary=summary,
            total_evaluation_time=time.time() - start_time
        )
    
    def _generate_evaluation_summary(
        self, 
        metric_scores: Dict[EvaluationMetric, EvaluationResult],
        evaluation_criteria: List[EvaluationCriteria]
    ) -> str:
        """Generate a summary of the evaluation results"""
        
        summary_parts = []
        
        for criteria in evaluation_criteria:
            if criteria.metric in metric_scores:
                result = metric_scores[criteria.metric]
                summary_parts.append(
                    f"{criteria.metric.value.title()}: {result.score:.2f} (weight: {criteria.weight:.1f})"
                )
        
        # Identify strengths and weaknesses
        strengths = [
            criteria.metric.value for criteria in evaluation_criteria
            if criteria.metric in metric_scores and metric_scores[criteria.metric].score >= 0.8
        ]
        
        weaknesses = [
            criteria.metric.value for criteria in evaluation_criteria
            if criteria.metric in metric_scores and metric_scores[criteria.metric].score < 0.6
        ]
        
        summary = "Evaluation Results:\n" + "\n".join(summary_parts)
        
        if strengths:
            summary += f"\n\nStrengths: {', '.join(strengths)}"
        
        if weaknesses:
            summary += f"\nAreas for improvement: {', '.join(weaknesses)}"
        
        return summary
    
    async def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]],
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None
    ) -> List[ComprehensiveEvaluation]:
        """Perform batch evaluation on multiple prompt-input-output combinations"""
        
        evaluation_tasks = []
        
        for eval_data in evaluations:
            task = self.comprehensive_evaluate(
                prompt=eval_data["prompt"],
                input_text=eval_data["input"],
                output=eval_data["output"],
                expected=eval_data.get("expected"),
                evaluation_criteria=evaluation_criteria
            )
            evaluation_tasks.append(task)
        
        return await asyncio.gather(*evaluation_tasks)
    
    def export_evaluation_results(self, evaluations: List[ComprehensiveEvaluation], filename: str):
        """Export evaluation results to JSON file"""
        export_data = {
            "evaluation_count": len(evaluations),
            "average_overall_score": np.mean([e.overall_score for e in evaluations]),
            "average_weighted_score": np.mean([e.weighted_score for e in evaluations]),
            "evaluations": [
                {
                    "overall_score": eval.overall_score,
                    "weighted_score": eval.weighted_score,
                    "evaluation_summary": eval.evaluation_summary,
                    "total_evaluation_time": eval.total_evaluation_time,
                    "metric_scores": {
                        metric.value: asdict(result) 
                        for metric, result in eval.metric_scores.items()
                    }
                }
                for eval in evaluations
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

class BenchmarkEvaluator:
    """Evaluator for benchmarking prompts against standard datasets"""
    
    def __init__(self, evaluation_system: ComprehensiveEvaluationSystem):
        self.evaluation_system = evaluation_system
        self.benchmark_results = {}
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        test_cases: List[Dict[str, Any]],
        prompts_to_test: List[str],
        evaluation_criteria: Optional[List[EvaluationCriteria]] = None
    ) -> Dict[str, Any]:
        """Run benchmark evaluation on multiple prompts"""
        
        benchmark_results = {}
        
        for i, prompt in enumerate(prompts_to_test):
            prompt_name = f"prompt_{i+1}"
            
            # Prepare evaluation data for this prompt
            evaluations = []
            for test_case in test_cases:
                evaluations.append({
                    "prompt": prompt,
                    "input": test_case["input"],
                    "output": test_case.get("output", ""),  # This would be generated
                    "expected": test_case.get("expected")
                })
            
            # Run batch evaluation
            results = await self.evaluation_system.batch_evaluate(evaluations, evaluation_criteria)
            
            # Calculate aggregate metrics
            aggregate_metrics = {
                "average_overall_score": np.mean([r.overall_score for r in results]),
                "average_weighted_score": np.mean([r.weighted_score for r in results]),
                "consistency": 1.0 - np.std([r.overall_score for r in results]),
                "total_evaluation_time": sum(r.total_evaluation_time for r in results)
            }
            
            benchmark_results[prompt_name] = {
                "prompt": prompt,
                "aggregate_metrics": aggregate_metrics,
                "detailed_results": results
            }
        
        self.benchmark_results[benchmark_name] = benchmark_results
        return benchmark_results
    
    def get_benchmark_summary(self, benchmark_name: str) -> Dict[str, Any]:
        """Get summary of benchmark results"""
        if benchmark_name not in self.benchmark_results:
            return {"error": "Benchmark not found"}
        
        results = self.benchmark_results[benchmark_name]
        
        # Find best performing prompt
        best_prompt = max(
            results.items(),
            key=lambda x: x[1]["aggregate_metrics"]["average_overall_score"]
        )
        
        return {
            "benchmark_name": benchmark_name,
            "prompts_tested": len(results),
            "best_prompt": best_prompt[0],
            "best_score": best_prompt[1]["aggregate_metrics"]["average_overall_score"],
            "score_range": {
                "min": min(r["aggregate_metrics"]["average_overall_score"] for r in results.values()),
                "max": max(r["aggregate_metrics"]["average_overall_score"] for r in results.values())
            }
        }