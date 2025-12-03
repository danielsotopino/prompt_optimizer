"""
Advanced Prompt Optimization Pipeline
Implements specialized optimization strategies and pipeline management
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from spo_framework import SPOFramework, PromptOptimizationConfig, OptimizationResult

class OptimizationStrategy(Enum):
    """Different optimization strategies available"""
    ITERATIVE_REFINEMENT = "iterative_refinement"
    MULTI_OBJECTIVE = "multi_objective" 
    ENSEMBLE_VOTING = "ensemble_voting"
    GENETIC_ALGORITHM = "genetic_algorithm"

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization"""
    objectives: List[str]  # e.g., ["accuracy", "clarity", "brevity"]
    weights: List[float]   # Weights for each objective
    
@dataclass
class PipelineConfig:
    """Configuration for the optimization pipeline"""
    strategy: OptimizationStrategy = OptimizationStrategy.ITERATIVE_REFINEMENT
    parallel_candidates: int = 3  # Number of parallel optimization candidates
    ensemble_size: int = 5        # Size of ensemble for voting
    genetic_population: int = 10  # Population size for genetic algorithm
    convergence_threshold: float = 0.01  # Threshold for convergence detection

class AdvancedOptimizationPipeline:
    """Advanced pipeline for prompt optimization with multiple strategies"""
    
    def __init__(self, spo_config: PromptOptimizationConfig, pipeline_config: PipelineConfig, api_key: str):
        self.spo_config = spo_config
        self.pipeline_config = pipeline_config
        self.api_key = api_key
        self.optimization_runs: List[Dict[str, Any]] = []
        
    async def optimize_with_strategy(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None,
        custom_evaluator: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize prompt using the specified strategy
        """
        strategy = self.pipeline_config.strategy
        
        if strategy == OptimizationStrategy.ITERATIVE_REFINEMENT:
            return await self._iterative_refinement_optimization(
                initial_prompt, task_description, sample_inputs, expected_outputs
            )
        elif strategy == OptimizationStrategy.MULTI_OBJECTIVE:
            return await self._multi_objective_optimization(
                initial_prompt, task_description, sample_inputs, expected_outputs
            )
        elif strategy == OptimizationStrategy.ENSEMBLE_VOTING:
            return await self._ensemble_voting_optimization(
                initial_prompt, task_description, sample_inputs, expected_outputs
            )
        elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
            return await self._genetic_algorithm_optimization(
                initial_prompt, task_description, sample_inputs, expected_outputs
            )
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    async def _iterative_refinement_optimization(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Standard iterative refinement optimization"""
        framework = SPOFramework(self.spo_config, self.api_key)
        result = await framework.optimize_prompt(
            initial_prompt, task_description, sample_inputs, expected_outputs
        )
        
        self.optimization_runs.append({
            "strategy": "iterative_refinement",
            "result": asdict(result),
            "summary": framework.get_optimization_summary()
        })
        
        return result
    
    async def _multi_objective_optimization(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Multi-objective optimization considering multiple criteria"""
        
        # Define multiple optimization objectives
        objectives = [
            ("accuracy", 0.4),
            ("clarity", 0.3),
            ("efficiency", 0.2),
            ("robustness", 0.1)
        ]
        
        best_results = []
        
        for objective, weight in objectives:
            # Create specialized framework for each objective
            specialized_config = PromptOptimizationConfig(
                max_iterations=3,
                optimization_model=self.spo_config.optimization_model,
                execution_model=self.spo_config.execution_model,
                evaluation_model=self.spo_config.evaluation_model
            )
            
            framework = SPOFramework(specialized_config, self.api_key)
            
            # Customize task description for specific objective
            objective_task = f"{task_description} (Focus on {objective})"
            
            result = await framework.optimize_prompt(
                initial_prompt, objective_task, sample_inputs, expected_outputs
            )
            
            best_results.append((result, weight))
        
        # Combine results using weighted voting
        combined_result = await self._combine_multi_objective_results(
            best_results, initial_prompt, task_description, sample_inputs, expected_outputs
        )
        
        self.optimization_runs.append({
            "strategy": "multi_objective",
            "result": asdict(combined_result),
            "objectives": [obj for obj, _ in objectives],
            "weights": [weight for _, weight in objectives]
        })
        
        return combined_result
    
    async def _ensemble_voting_optimization(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Ensemble optimization using multiple parallel optimization runs"""
        
        ensemble_size = self.pipeline_config.ensemble_size
        optimization_tasks = []
        
        # Create multiple optimization instances with slight variations
        for i in range(ensemble_size):
            config = PromptOptimizationConfig(
                max_iterations=self.spo_config.max_iterations,
                optimization_model=self.spo_config.optimization_model,
                execution_model=self.spo_config.execution_model,
                evaluation_model=self.spo_config.evaluation_model,
                temperature=self.spo_config.temperature + (i * 0.1)  # Slight temperature variation
            )
            
            framework = SPOFramework(config, self.api_key)
            task = framework.optimize_prompt(
                initial_prompt, task_description, sample_inputs, expected_outputs
            )
            optimization_tasks.append(task)
        
        # Run all optimizations in parallel
        ensemble_results = await asyncio.gather(*optimization_tasks)
        
        # Select best result based on performance scores
        best_result = max(ensemble_results, key=lambda x: x.performance_score)
        
        # Create ensemble summary
        ensemble_summary = {
            "ensemble_size": ensemble_size,
            "individual_scores": [r.performance_score for r in ensemble_results],
            "best_score": best_result.performance_score,
            "score_variance": np.var([r.performance_score for r in ensemble_results]),
            "consensus_level": self._calculate_consensus_level(ensemble_results)
        }
        
        self.optimization_runs.append({
            "strategy": "ensemble_voting",
            "result": asdict(best_result),
            "ensemble_summary": ensemble_summary
        })
        
        return best_result
    
    async def _genetic_algorithm_optimization(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Genetic algorithm-inspired optimization"""
        
        population_size = self.pipeline_config.genetic_population
        generations = 3
        
        # Initialize population with variations of the initial prompt
        population = await self._generate_initial_population(
            initial_prompt, task_description, population_size
        )
        
        best_overall = None
        
        for generation in range(generations):
            # Evaluate each individual in the population
            evaluated_population = []
            
            for individual_prompt in population:
                framework = SPOFramework(self.spo_config, self.api_key)
                result = await framework.optimize_prompt(
                    individual_prompt, task_description, sample_inputs, expected_outputs
                )
                evaluated_population.append((individual_prompt, result))
                
                if best_overall is None or result.performance_score > best_overall.performance_score:
                    best_overall = result
            
            # Selection: Keep top 50% performers
            evaluated_population.sort(key=lambda x: x[1].performance_score, reverse=True)
            top_performers = evaluated_population[:population_size // 2]
            
            # Crossover and mutation: Generate new population
            if generation < generations - 1:
                population = await self._generate_next_generation(
                    [p[0] for p in top_performers], task_description
                )
        
        self.optimization_runs.append({
            "strategy": "genetic_algorithm",
            "result": asdict(best_overall),
            "generations": generations,
            "population_size": population_size
        })
        
        return best_overall
    
    async def _generate_initial_population(
        self, 
        initial_prompt: str, 
        task_description: str, 
        population_size: int
    ) -> List[str]:
        """Generate initial population for genetic algorithm"""
        
        variation_prompts = [
            f"Create {population_size} different variations of this prompt for the task: {task_description}\n\nOriginal prompt: {initial_prompt}\n\nReturn only the variations, one per line."
        ]
        
        framework = SPOFramework(self.spo_config, self.api_key)
        
        try:
            response = framework.client.chat_completions_create(
                model=self.spo_config.optimization_model,
                messages=[
                    {"role": "user", "content": variation_prompts[0]}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            # Ensure we have enough variations
            while len(variations) < population_size:
                variations.append(initial_prompt)
            
            return variations[:population_size]
            
        except Exception:
            # Fallback: return copies of original prompt
            return [initial_prompt] * population_size
    
    async def _generate_next_generation(
        self, 
        top_performers: List[str], 
        task_description: str
    ) -> List[str]:
        """Generate next generation through crossover and mutation"""
        
        next_generation = []
        target_size = self.pipeline_config.genetic_population
        
        # Keep the best performers
        next_generation.extend(top_performers)
        
        # Generate new individuals through crossover
        while len(next_generation) < target_size:
            # Select two random parents
            import random
            parent1 = random.choice(top_performers)
            parent2 = random.choice(top_performers)
            
            # Create offspring through combination
            crossover_prompt = f"""Combine these two prompts for the task: {task_description}

Prompt 1: {parent1}

Prompt 2: {parent2}

Create a new prompt that combines the best aspects of both. Return only the new prompt."""

            try:
                framework = SPOFramework(self.spo_config, self.api_key)
                response = framework.client.chat_completions_create(
                    model=self.spo_config.optimization_model,
                    messages=[
                        {"role": "user", "content": crossover_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                offspring = response.choices[0].message.content.strip()
                next_generation.append(offspring)
                
            except Exception:
                # Fallback: use one of the parents
                next_generation.append(parent1)
        
        return next_generation[:target_size]
    
    async def _combine_multi_objective_results(
        self,
        weighted_results: List[tuple],
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> OptimizationResult:
        """Combine results from multi-objective optimization"""
        
        # Calculate weighted scores
        best_weighted_score = 0
        best_result = None
        
        for result, weight in weighted_results:
            weighted_score = result.performance_score * weight
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_result = result
        
        return best_result
    
    def _calculate_consensus_level(self, results: List[OptimizationResult]) -> float:
        """Calculate consensus level among ensemble results"""
        scores = [r.performance_score for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Consensus is higher when standard deviation is lower
        consensus = max(0.0, 1.0 - (std_score / mean_score if mean_score > 0 else 1.0))
        return consensus
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs"""
        return {
            "total_runs": len(self.optimization_runs),
            "strategies_used": list(set(run["strategy"] for run in self.optimization_runs)),
            "best_overall_score": max(
                run["result"]["performance_score"] for run in self.optimization_runs
            ) if self.optimization_runs else 0,
            "runs": self.optimization_runs
        }
    
    def export_pipeline_results(self, filename: str):
        """Export all pipeline results to JSON file"""
        pipeline_data = {
            "pipeline_config": asdict(self.pipeline_config),
            "spo_config": asdict(self.spo_config),
            "summary": self.get_pipeline_summary(),
            "detailed_runs": self.optimization_runs
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pipeline_data, f, indent=2, ensure_ascii=False)

class OptimizationOrchestrator:
    """High-level orchestrator for running multiple optimization strategies"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results_comparison: Dict[str, Any] = {}
    
    async def compare_strategies(
        self,
        initial_prompt: str,
        task_description: str,
        sample_inputs: List[str],
        expected_outputs: Optional[List[str]] = None,
        strategies: Optional[List[OptimizationStrategy]] = None
    ) -> Dict[str, OptimizationResult]:
        """Compare multiple optimization strategies"""
        
        if strategies is None:
            strategies = list(OptimizationStrategy)
        
        spo_config = PromptOptimizationConfig(max_iterations=3)
        results = {}
        
        for strategy in strategies:
            pipeline_config = PipelineConfig(strategy=strategy)
            pipeline = AdvancedOptimizationPipeline(spo_config, pipeline_config, self.api_key)
            
            try:
                result = await pipeline.optimize_with_strategy(
                    initial_prompt, task_description, sample_inputs, expected_outputs
                )
                results[strategy.value] = result
            except Exception as e:
                print(f"Error with strategy {strategy.value}: {str(e)}")
                continue
        
        self.results_comparison = {
            "strategies_compared": [s.value for s in strategies],
            "results": {k: asdict(v) for k, v in results.items()},
            "best_strategy": max(results.items(), key=lambda x: x[1].performance_score)[0] if results else None
        }
        
        return results
    
    def get_strategy_recommendations(self) -> Dict[str, str]:
        """Get recommendations for when to use each strategy"""
        return {
            "iterative_refinement": "Best for: General purpose optimization, when you have clear evaluation criteria",
            "multi_objective": "Best for: When you need to balance multiple competing objectives (accuracy vs brevity)",
            "ensemble_voting": "Best for: When you want robust results and can afford computational cost",
            "genetic_algorithm": "Best for: Exploring diverse prompt variations, creative optimization tasks"
        }