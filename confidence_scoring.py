"""
Confidence Scoring System
Based on MetaGPT notebook approach for measuring optimization confidence
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from llm_client import LLMClient

@dataclass
class ConfidenceMetrics:
    """Confidence metrics for optimization results"""
    consistency_score: float  # How consistent are results across multiple runs
    convergence_confidence: float  # How confident are we in convergence
    stability_score: float  # How stable is the performance
    reliability_index: float  # Overall reliability measure
    variance_penalty: float  # Penalty for high variance

class ConfidenceAnalyzer:
    """Analyzer for measuring confidence in optimization results"""
    
    def __init__(self, client: LLMClient, evaluation_model: str = "gpt-4o"):
        self.client = client
        self.evaluation_model = evaluation_model
    
    async def analyze_optimization_confidence(
        self,
        optimization_history: List[Dict[str, Any]],
        test_inputs: List[str],
        final_prompt: str,
        num_validation_runs: int = 5
    ) -> ConfidenceMetrics:
        """
        Analyze confidence in optimization results using multiple validation approaches
        """
        
        # 1. Consistency Analysis - run final prompt multiple times
        consistency_scores = await self._measure_consistency(
            final_prompt, test_inputs, num_validation_runs
        )
        consistency_score = 1.0 - np.std(consistency_scores)  # Lower std = higher consistency
        
        # 2. Convergence Confidence - analyze optimization trajectory
        convergence_confidence = self._analyze_convergence(optimization_history)
        
        # 3. Stability Score - measure performance stability
        stability_score = self._measure_stability(optimization_history)
        
        # 4. Variance Penalty - penalize high variance in results
        variance_penalty = self._calculate_variance_penalty(consistency_scores)
        
        # 5. Overall Reliability Index
        reliability_index = (
            consistency_score * 0.3 +
            convergence_confidence * 0.3 +
            stability_score * 0.3 +
            (1.0 - variance_penalty) * 0.1
        )
        
        return ConfidenceMetrics(
            consistency_score=consistency_score,
            convergence_confidence=convergence_confidence,
            stability_score=stability_score,
            reliability_index=reliability_index,
            variance_penalty=variance_penalty
        )
    
    async def _measure_consistency(
        self, 
        prompt: str, 
        test_inputs: List[str], 
        num_runs: int
    ) -> List[float]:
        """Measure consistency by running the same prompt multiple times"""
        
        all_run_scores = []
        
        for run in range(num_runs):
            run_scores = []
            
            for test_input in test_inputs:
                try:
                    # Execute prompt
                    response = self.client.chat_completions_create(
                        model=self.evaluation_model,
                        messages=[{"role": "user", "content": f"{prompt}\n\nInput: {test_input}"}],
                        temperature=0.1,  # Low temperature for consistency
                        max_tokens=500
                    )
                    
                    output = response.choices[0].message.content.strip()
                    
                    # Score quality (simplified - in practice you'd use your evaluation system)
                    quality_score = await self._quick_quality_score(output, test_input)
                    run_scores.append(quality_score)
                    
                except Exception as e:
                    run_scores.append(0.0)  # Failure penalty
            
            all_run_scores.append(np.mean(run_scores))
        
        return all_run_scores
    
    def _analyze_convergence(self, optimization_history: List[Dict[str, Any]]) -> float:
        """Analyze if optimization converged properly"""
        
        if len(optimization_history) < 2:
            return 0.5  # Not enough data
        
        scores = []
        for entry in optimization_history:
            if isinstance(entry, dict):
                score = entry.get('performance_score', 0) or entry.get('score', 0)
                scores.append(score)
        
        if len(scores) < 2:
            return 0.5
        
        # Check for improvement trend
        improvements = []
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i-1]
            improvements.append(improvement)
        
        # Convergence indicators
        final_improvement = abs(improvements[-1]) if improvements else 1.0
        trend_consistency = 1.0 - np.std(improvements) if len(improvements) > 1 else 0.5
        overall_improvement = scores[-1] - scores[0] if scores else 0.0
        
        # Combine indicators
        convergence_confidence = (
            (1.0 - final_improvement) * 0.4 +  # Small final changes = good
            trend_consistency * 0.3 +          # Consistent trend = good
            min(overall_improvement * 2, 1.0) * 0.3  # Overall improvement = good
        )
        
        return max(0.0, min(1.0, convergence_confidence))
    
    def _measure_stability(self, optimization_history: List[Dict[str, Any]]) -> float:
        """Measure stability of optimization performance"""
        
        scores = []
        for entry in optimization_history:
            if isinstance(entry, dict):
                score = entry.get('performance_score', 0) or entry.get('score', 0)
                scores.append(score)
        
        if len(scores) < 2:
            return 0.5
        
        # Calculate stability metrics
        score_variance = np.var(scores)
        score_range = max(scores) - min(scores)
        
        # Stability is inverse of variance and range
        stability_score = 1.0 / (1.0 + score_variance + score_range)
        
        return min(1.0, stability_score)
    
    def _calculate_variance_penalty(self, consistency_scores: List[float]) -> float:
        """Calculate penalty for high variance in consistency scores"""
        
        if len(consistency_scores) < 2:
            return 0.0
        
        variance = np.var(consistency_scores)
        
        # Normalize variance to 0-1 scale (penalty)
        # High variance = high penalty
        variance_penalty = min(1.0, variance * 5)  # Scale factor of 5
        
        return variance_penalty
    
    async def _quick_quality_score(self, output: str, input_text: str) -> float:
        """Quick quality scoring for consistency measurement"""
        
        # Simple heuristics for quick scoring
        score = 0.5  # Base score
        
        # Length check
        if 10 <= len(output) <= 200:
            score += 0.2
        
        # Contains expected format elements
        if " - " in output:  # For job classification
            score += 0.2
        
        # Not empty or error-like
        if output and "error" not in output.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def generate_confidence_report(
        self, 
        metrics: ConfidenceMetrics,
        optimization_type: str = "Unknown"
    ) -> Dict[str, Any]:
        """Generate a comprehensive confidence report"""
        
        # Determine confidence level
        reliability = metrics.reliability_index
        
        if reliability >= 0.8:
            confidence_level = "High"
            recommendation = "Results are highly reliable. Safe for production use."
        elif reliability >= 0.6:
            confidence_level = "Medium"
            recommendation = "Results are moderately reliable. Consider additional validation."
        elif reliability >= 0.4:
            confidence_level = "Low"
            recommendation = "Results have low reliability. Further optimization recommended."
        else:
            confidence_level = "Very Low"
            recommendation = "Results are unreliable. Significant improvements needed."
        
        # Identify specific issues
        issues = []
        if metrics.consistency_score < 0.6:
            issues.append("Low consistency across multiple runs")
        if metrics.convergence_confidence < 0.6:
            issues.append("Poor convergence during optimization")
        if metrics.stability_score < 0.6:
            issues.append("Unstable performance during optimization")
        if metrics.variance_penalty > 0.3:
            issues.append("High variance in results")
        
        return {
            "confidence_level": confidence_level,
            "reliability_index": reliability,
            "detailed_metrics": {
                "consistency_score": metrics.consistency_score,
                "convergence_confidence": metrics.convergence_confidence,
                "stability_score": metrics.stability_score,
                "variance_penalty": metrics.variance_penalty
            },
            "issues_identified": issues,
            "recommendation": recommendation,
            "optimization_type": optimization_type,
            "confidence_breakdown": {
                "consistency": f"{metrics.consistency_score:.2f} - {'Good' if metrics.consistency_score > 0.7 else 'Needs improvement'}",
                "convergence": f"{metrics.convergence_confidence:.2f} - {'Good' if metrics.convergence_confidence > 0.7 else 'Needs improvement'}",
                "stability": f"{metrics.stability_score:.2f} - {'Good' if metrics.stability_score > 0.7 else 'Needs improvement'}"
            }
        }

class AutoStoppingCriteria:
    """Automatic stopping criteria for optimization based on confidence metrics"""
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.8,
        max_iterations_without_improvement: int = 3,
        convergence_threshold: float = 0.01
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.convergence_threshold = convergence_threshold
        self.iteration_history = []
    
    def should_stop_optimization(
        self,
        current_score: float,
        confidence_metrics: ConfidenceMetrics,
        iteration: int
    ) -> Tuple[bool, str]:
        """
        Determine if optimization should stop based on confidence and performance
        """
        
        self.iteration_history.append({
            'iteration': iteration,
            'score': current_score,
            'confidence': confidence_metrics.reliability_index
        })
        
        # Check confidence threshold
        if confidence_metrics.reliability_index >= self.min_confidence_threshold:
            return True, f"High confidence achieved ({confidence_metrics.reliability_index:.2f})"
        
        # Check for convergence
        if len(self.iteration_history) >= 2:
            recent_improvements = []
            for i in range(max(1, len(self.iteration_history) - self.max_iterations_without_improvement), 
                          len(self.iteration_history)):
                if i > 0:
                    improvement = self.iteration_history[i]['score'] - self.iteration_history[i-1]['score']
                    recent_improvements.append(improvement)
            
            if recent_improvements and max(recent_improvements) < self.convergence_threshold:
                return True, f"Convergence detected (max improvement: {max(recent_improvements):.4f})"
        
        # Check for stability in confidence
        if len(self.iteration_history) >= 3:
            recent_confidences = [h['confidence'] for h in self.iteration_history[-3:]]
            if np.std(recent_confidences) < 0.05 and np.mean(recent_confidences) > 0.7:
                return True, f"Stable high confidence achieved"
        
        return False, "Continue optimization"
    
    def get_stopping_summary(self) -> Dict[str, Any]:
        """Get summary of stopping criteria analysis"""
        
        if not self.iteration_history:
            return {"status": "No data available"}
        
        final_entry = self.iteration_history[-1]
        
        return {
            "total_iterations": len(self.iteration_history),
            "final_score": final_entry['score'],
            "final_confidence": final_entry['confidence'],
            "score_improvement": final_entry['score'] - self.iteration_history[0]['score'] if len(self.iteration_history) > 1 else 0,
            "confidence_trend": [h['confidence'] for h in self.iteration_history],
            "score_trend": [h['score'] for h in self.iteration_history]
        }