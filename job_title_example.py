"""
Job Title Classification Example
Demonstrates the SPO framework with the job title classification use case from the notebook
"""
import asyncio
import os
from typing import List, Dict, Any
from spo_framework import SPOFramework, PromptOptimizationConfig
from optimization_pipeline import AdvancedOptimizationPipeline, PipelineConfig, OptimizationStrategy
from evaluation_system import ComprehensiveEvaluationSystem, EvaluationCriteria, EvaluationMetric
from lambda_functions import MetaGPTStyleOptimizer, PromptLambdaFactory

class JobTitleClassificationExample:
    """Example implementation of job title classification optimization"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sample_job_titles = [
            "Senior Software Engineer at Google",
            "Data Scientist - Machine Learning",
            "Marketing Manager, Digital Products",
            "Chief Technology Officer",
            "Junior Frontend Developer",
            "Product Manager - AI/ML",
            "Senior DevOps Engineer",
            "UX/UI Designer",
            "Sales Director, Enterprise Solutions",
            "Business Analyst - Financial Services"
        ]
        
        self.expected_classifications = [
            "Technology - Software Engineering",
            "Technology - Data Science",
            "Marketing - Digital Marketing",
            "Executive - Technology",
            "Technology - Software Engineering",
            "Product - AI/ML",
            "Technology - Infrastructure",
            "Design - User Experience",
            "Sales - Enterprise",
            "Business - Financial Analysis"
        ]
        
        self.initial_prompt = """Classify the following job title into appropriate categories. 
        Use the format: Category - Subcategory
        
        Job Title: {job_title}
        Classification:"""
        
        # Enhanced prompt based on MetaGPT notebook optimization
        self.enhanced_prompt = """You are an expert HR professional specializing in job classification. Classify the following job title into appropriate categories and subcategories.

**Classification Guidelines:**
- Technology: Software Engineering, Data Science, DevOps, Security, AI/ML, Infrastructure
- Business: Sales, Marketing, Operations, Strategy, Finance, HR
- Executive: C-Level, VP-Level, Director-Level
- Creative: Design, Content, Marketing Creative
- Operations: Manufacturing, Logistics, Quality, Process

**Examples:**
- "Senior Software Engineer" → "Technology - Software Engineering"
- "VP of Sales" → "Executive - Sales"
- "Data Scientist - Machine Learning" → "Technology - Data Science"
- "Marketing Manager" → "Business - Marketing"

**Special Cases:**
- For ambiguous titles, choose the most likely primary function
- For hybrid roles, prioritize the main responsibility
- For consultant roles, focus on the domain expertise
- For startup roles with multiple responsibilities, classify by primary skill

**Required Format:** Category - Subcategory

Job Title: {job_title}

Classification:"""
        
        self.task_description = "Classify job titles into meaningful categories and subcategories"
    
    async def run_basic_optimization(self) -> Dict[str, Any]:
        """Run basic SPO optimization on job title classification"""
        
        print("🚀 Running basic SPO optimization for job title classification...")
        
        # Configure optimization
        config = PromptOptimizationConfig(
            max_iterations=5,
            optimization_model="gpt-4o",
            execution_model="gpt-4o-mini",
            evaluation_model="gpt-4o",
            temperature=0.7
        )
        
        # Create framework
        framework = SPOFramework(config, self.api_key)
        
        # Run optimization
        result = await framework.optimize_prompt(
            initial_prompt=self.initial_prompt,
            task_description=self.task_description,
            sample_inputs=self.sample_job_titles,
            expected_outputs=self.expected_classifications
        )
        
        print(f"✅ Basic optimization completed!")
        print(f"📈 Performance improvement: {result.performance_score:.2f}")
        print(f"🔄 Iterations: {result.iteration}")
        
        return {
            "optimization_type": "basic_spo",
            "result": result,
            "summary": framework.get_optimization_summary()
        }
    
    async def run_advanced_pipeline_comparison(self) -> Dict[str, Any]:
        """Compare different optimization strategies"""
        
        print("🔬 Running advanced pipeline comparison...")
        
        strategies_to_test = [
            OptimizationStrategy.ITERATIVE_REFINEMENT,
            OptimizationStrategy.ENSEMBLE_VOTING,
            OptimizationStrategy.MULTI_OBJECTIVE
        ]
        
        results = {}
        
        for strategy in strategies_to_test:
            print(f"🧪 Testing strategy: {strategy.value}")
            
            # Configure pipeline
            spo_config = PromptOptimizationConfig(max_iterations=3)
            pipeline_config = PipelineConfig(
                strategy=strategy,
                parallel_candidates=3,
                ensemble_size=3 if strategy == OptimizationStrategy.ENSEMBLE_VOTING else 5
            )
            
            # Run optimization
            pipeline = AdvancedOptimizationPipeline(spo_config, pipeline_config, self.api_key)
            
            try:
                result = await pipeline.optimize_with_strategy(
                    initial_prompt=self.initial_prompt,
                    task_description=self.task_description,
                    sample_inputs=self.sample_job_titles[:5],  # Use subset for faster testing
                    expected_outputs=self.expected_classifications[:5]
                )
                
                results[strategy.value] = {
                    "result": result,
                    "strategy_config": pipeline_config
                }
                
                print(f"✅ {strategy.value}: Score {result.performance_score:.2f}")
                
            except Exception as e:
                print(f"❌ Error with {strategy.value}: {str(e)}")
                continue
        
        # Find best strategy
        if results:
            best_strategy = max(results.items(), key=lambda x: x[1]["result"].performance_score)
            print(f"🏆 Best strategy: {best_strategy[0]} (Score: {best_strategy[1]['result'].performance_score:.2f})")
        
        return results
    
    async def run_metagpt_style_optimization(self) -> Dict[str, Any]:
        """Run MetaGPT notebook style optimization using lambda functions"""
        
        print("🧪 Running MetaGPT-style lambda optimization...")
        
        # Configure optimization
        config = PromptOptimizationConfig(
            max_iterations=5,
            optimization_model="gpt-4o",
            execution_model="gpt-4o-mini",
            evaluation_model="gpt-4o"
        )
        
        # Create MetaGPT style optimizer
        from llm_client import create_llm_client
        client = create_llm_client(api_key=self.api_key)
        optimizer = MetaGPTStyleOptimizer(client, config)
        
        # Create initial lambda function
        initial_lambda = PromptLambdaFactory.create_job_classification_lambda()
        
        # Run optimization
        result = await optimizer.optimize_with_lambda(
            initial_lambda=initial_lambda,
            task_description=self.task_description,
            test_inputs=self.sample_job_titles,
            expected_outputs=self.expected_classifications,
            max_rounds=5
        )
        
        print(f"✅ MetaGPT-style optimization completed!")
        print(f"📈 Best score: {result['best_score']:.2f}")
        print(f"🔄 Rounds completed: {len(result['optimization_history'])}")
        
        return {
            "optimization_type": "metagpt_lambda_style",
            "result": result,
            "best_prompt": result['final_prompt_template']
        }
    
    async def run_comprehensive_evaluation(self, optimized_prompt: str) -> Dict[str, Any]:
        """Run comprehensive evaluation on the optimized prompt"""
        
        print("📊 Running comprehensive evaluation...")
        
        # Setup evaluation system
        evaluation_system = ComprehensiveEvaluationSystem(self.api_key)
        
        # Define evaluation criteria
        criteria = [
            EvaluationCriteria(EvaluationMetric.ACCURACY, 0.5, "Correctness of classification"),
            EvaluationCriteria(EvaluationMetric.RELEVANCE, 0.2, "Relevance to job title"),
            EvaluationCriteria(EvaluationMetric.CLARITY, 0.2, "Clear classification format"),
            EvaluationCriteria(EvaluationMetric.ADHERENCE_TO_FORMAT, 0.1, "Follows required format")
        ]
        
        # Add format evaluator
        evaluation_system.add_format_evaluator("Category - Subcategory")
        
        # Prepare evaluation data
        evaluations = []
        for i, job_title in enumerate(self.sample_job_titles):
            # Generate output using optimized prompt
            formatted_prompt = optimized_prompt.format(job_title=job_title)
            
            # For demo purposes, we'll simulate the output
            # In real usage, you'd call the LLM here
            simulated_output = self.expected_classifications[i]  # Using expected as simulation
            
            evaluations.append({
                "prompt": formatted_prompt,
                "input": job_title,
                "output": simulated_output,
                "expected": self.expected_classifications[i]
            })
        
        # Run batch evaluation
        results = await evaluation_system.batch_evaluate(evaluations, criteria)
        
        # Generate summary
        avg_score = sum(r.overall_score for r in results) / len(results)
        avg_weighted_score = sum(r.weighted_score for r in results) / len(results)
        
        print(f"📈 Average Overall Score: {avg_score:.2f}")
        print(f"⚖️ Average Weighted Score: {avg_weighted_score:.2f}")
        
        return {
            "evaluation_results": results,
            "average_overall_score": avg_score,
            "average_weighted_score": avg_weighted_score,
            "criteria_used": criteria
        }
    
    async def run_complete_example(self) -> Dict[str, Any]:
        """Run the complete job title classification example"""
        
        print("🎯 Starting complete job title classification optimization example")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Basic optimization
        print("\n📝 Step 1: Basic SPO Optimization")
        basic_result = await self.run_basic_optimization()
        results["basic_optimization"] = basic_result
        
        # Step 2: MetaGPT-style optimization
        print("\n🧪 Step 2: MetaGPT-style Lambda Optimization")
        metagpt_results = await self.run_metagpt_style_optimization()
        results["metagpt_optimization"] = metagpt_results
        
        # Step 3: Advanced pipeline comparison
        print("\n🔄 Step 3: Advanced Pipeline Comparison")
        pipeline_results = await self.run_advanced_pipeline_comparison()
        results["pipeline_comparison"] = pipeline_results
        
        # Step 4: Comprehensive evaluation
        print("\n📊 Step 4: Comprehensive Evaluation")
        best_prompt = basic_result["result"].optimized_prompt
        evaluation_results = await self.run_comprehensive_evaluation(best_prompt)
        results["comprehensive_evaluation"] = evaluation_results
        
        # Step 5: Generate final report
        print("\n📋 Step 5: Final Report")
        final_report = self.generate_final_report(results)
        results["final_report"] = final_report
        
        print("\n🎉 Complete example finished!")
        print("=" * 60)
        
        return results
    
    def generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive final report"""
        
        basic_score = results["basic_optimization"]["result"].performance_score
        metagpt_score = results.get("metagpt_optimization", {}).get("result", {}).get("best_score", 0)
        
        # Find best pipeline strategy
        pipeline_scores = {}
        for strategy, data in results.get("pipeline_comparison", {}).items():
            pipeline_scores[strategy] = data["result"].performance_score
        
        best_pipeline = max(pipeline_scores.items(), key=lambda x: x[1]) if pipeline_scores else None
        
        # Evaluation summary
        eval_score = results.get("comprehensive_evaluation", {}).get("average_overall_score", 0)
        
        report = {
            "summary": {
                "task": "Job Title Classification Optimization",
                "initial_prompt_length": len(self.initial_prompt),
                "sample_size": len(self.sample_job_titles),
                "optimization_methods_tested": 1 + len(results.get("pipeline_comparison", {}))
            },
            "performance_metrics": {
                "basic_spo_score": basic_score,
                "metagpt_lambda_score": metagpt_score,
                "best_pipeline_strategy": best_pipeline[0] if best_pipeline else None,
                "best_pipeline_score": best_pipeline[1] if best_pipeline else None,
                "comprehensive_evaluation_score": eval_score
            },
            "recommendations": self.generate_recommendations(results),
            "next_steps": [
                "Test with larger dataset",
                "Implement A/B testing in production",
                "Monitor performance over time",
                "Consider domain-specific fine-tuning"
            ]
        }
        
        # Print report summary
        print(f"📈 Basic SPO Score: {basic_score:.2f}")
        print(f"🧪 MetaGPT Lambda Score: {metagpt_score:.2f}")
        if best_pipeline:
            print(f"🏆 Best Pipeline: {best_pipeline[0]} (Score: {best_pipeline[1]:.2f})")
        print(f"📊 Evaluation Score: {eval_score:.2f}")
        
        return report
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        basic_score = results["basic_optimization"]["result"].performance_score
        
        if basic_score < 0.7:
            recommendations.append("Consider providing more examples in the prompt")
            recommendations.append("Refine the classification categories")
        elif basic_score < 0.8:
            recommendations.append("Add edge case handling to the prompt")
            recommendations.append("Consider multi-shot prompting")
        else:
            recommendations.append("Excellent performance! Consider deployment")
        
        # Pipeline-specific recommendations
        pipeline_results = results.get("pipeline_comparison", {})
        if pipeline_results:
            best_strategy = max(pipeline_results.items(), key=lambda x: x[1]["result"].performance_score)[0]
            recommendations.append(f"Use {best_strategy} strategy for production")
        
        recommendations.append("Monitor performance with real-world data")
        recommendations.append("Set up automated retraining pipeline")
        
        return recommendations

# Utility functions for easy usage
async def quick_job_title_demo(api_key: str):
    """Quick demo of job title classification optimization"""
    example = JobTitleClassificationExample(api_key)
    return await example.run_basic_optimization()

async def full_job_title_demo(api_key: str):
    """Full demo with all features"""
    example = JobTitleClassificationExample(api_key)
    return await example.run_complete_example()

# Main execution
if __name__ == "__main__":
    # You would set your OpenAI API key here
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not API_KEY:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    print("🤖 Job Title Classification Optimization Demo")
    print("Choose an option:")
    print("1. Quick demo (basic optimization only)")
    print("2. Full demo (all features)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    async def main():
        if choice == "1":
            result = await quick_job_title_demo(API_KEY)
            print("\n📊 Quick Demo Results:")
            print(f"Score: {result['result'].performance_score:.2f}")
            print(f"Optimized Prompt: {result['result'].optimized_prompt}")
        elif choice == "2":
            results = await full_job_title_demo(API_KEY)
            print("\n📊 Full Demo completed! Check results above.")
        else:
            print("❌ Invalid choice")
    
    asyncio.run(main())