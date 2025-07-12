"""
Main entry point for the SPO Framework
Provides CLI interface and example usage
"""
import asyncio
import argparse
import json
import os
from typing import List, Optional
from dotenv import load_dotenv
from spo_framework import SPOFramework, PromptOptimizationConfig
from optimization_pipeline import AdvancedOptimizationPipeline, PipelineConfig, OptimizationStrategy, OptimizationOrchestrator
from evaluation_system import ComprehensiveEvaluationSystem, EvaluationCriteria, EvaluationMetric
from job_title_example import JobTitleClassificationExample
from yaml_parser import YAMLParser

# Load environment variables from .env file
load_dotenv()

def setup_cli():
    """Setup command line interface"""
    parser = argparse.ArgumentParser(description="Self-Supervised Prompt Optimization Framework")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Basic optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Run basic prompt optimization')
    optimize_parser.add_argument('--prompt', help='Initial prompt to optimize (optional if included in YAML)')
    optimize_parser.add_argument('--task', help='Task description (optional if included in YAML)')
    optimize_parser.add_argument('--inputs', required=True, help='JSON or YAML file with sample inputs')
    optimize_parser.add_argument('--expected', help='JSON or YAML file with expected outputs')
    optimize_parser.add_argument('--iterations', type=int, default=5, help='Number of optimization iterations')
    optimize_parser.add_argument('--output', help='Output file for results')
    
    # Pipeline comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare optimization strategies')
    compare_parser.add_argument('--prompt', required=True, help='Initial prompt to optimize')
    compare_parser.add_argument('--task', required=True, help='Task description')
    compare_parser.add_argument('--inputs', required=True, help='JSON or YAML file with sample inputs')
    compare_parser.add_argument('--expected', help='JSON or YAML file with expected outputs')
    compare_parser.add_argument('--strategies', nargs='+', 
                               choices=['iterative_refinement', 'multi_objective', 'ensemble_voting', 'genetic_algorithm'],
                               default=['iterative_refinement', 'ensemble_voting'],
                               help='Strategies to compare')
    compare_parser.add_argument('--output', help='Output file for results')
    
    # Job title example command
    example_parser = subparsers.add_parser('job-title-example', help='Run job title classification example')
    example_parser.add_argument('--mode', choices=['quick', 'full'], default='quick', 
                               help='Demo mode: quick or full')
    example_parser.add_argument('--output', help='Output file for results')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a prompt')
    eval_parser.add_argument('--prompt', required=True, help='Prompt to evaluate')
    eval_parser.add_argument('--inputs', required=True, help='JSON or YAML file with test inputs')
    eval_parser.add_argument('--outputs', required=True, help='JSON or YAML file with outputs to evaluate')
    eval_parser.add_argument('--expected', help='JSON or YAML file with expected outputs')
    eval_parser.add_argument('--metrics', nargs='+',
                           choices=['accuracy', 'relevance', 'clarity', 'completeness', 'consistency'],
                           default=['accuracy', 'relevance', 'clarity'],
                           help='Evaluation metrics to use')
    eval_parser.add_argument('--output', help='Output file for results')
    
    return parser

async def run_basic_optimization(args):
    """Run basic SPO optimization"""
    print("🚀 Starting basic prompt optimization...")
    
    # Try to load configuration from YAML if prompt/task not provided
    prompt = args.prompt
    task = args.task
    
    if not prompt or not task:
        try:
            yaml_config = YAMLParser.parse_prompt_config(args.inputs)
            if not prompt and 'prompt' in yaml_config:
                prompt = yaml_config['prompt']
            if not task and 'task' in yaml_config:
                task = yaml_config['task']
        except Exception as e:
            pass
    
    # Validate required parameters
    if not prompt:
        print("❌ Error: --prompt is required or must be included in YAML file")
        return
    if not task:
        print("❌ Error: --task is required or must be included in YAML file")
        return
    
    # Load inputs
    sample_inputs = YAMLParser.load_inputs_flexible(args.inputs)
    
    expected_outputs = None
    if args.expected:
        expected_outputs = YAMLParser.load_expected_flexible(args.expected)
    else:
        # Try to get expected outputs from the same YAML file
        try:
            expected_outputs = YAMLParser.load_expected_flexible(args.inputs)
        except:
            pass
    
    # Configure optimization
    config = PromptOptimizationConfig(
        max_iterations=args.iterations,
        optimization_model="gpt-4o",
        execution_model="gpt-4o-mini",
        evaluation_model="gpt-4o"
    )
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Run optimization
    framework = SPOFramework(config, api_key)
    result = await framework.optimize_prompt(
        initial_prompt=prompt,
        task_description=task,
        sample_inputs=sample_inputs,
        expected_outputs=expected_outputs
    )
    
    # Output results
    print(f"✅ Optimization completed!")
    print(f"📈 Final score: {result.performance_score:.2f}")
    print(f"🔄 Iterations: {result.iteration}")
    print(f"📝 Optimized prompt:\n{result.optimized_prompt}")
    
    if args.output:
        framework.export_results(args.output)
        print(f"💾 Results saved to {args.output}")

async def run_strategy_comparison(args):
    """Run strategy comparison"""
    print("🔬 Starting strategy comparison...")
    
    # Load inputs
    sample_inputs = YAMLParser.load_inputs_flexible(args.inputs)
    
    expected_outputs = None
    if args.expected:
        expected_outputs = YAMLParser.load_expected_flexible(args.expected)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Convert strategy names to enum values
    strategies = [OptimizationStrategy(strategy) for strategy in args.strategies]
    
    # Run comparison
    orchestrator = OptimizationOrchestrator(api_key)
    results = await orchestrator.compare_strategies(
        initial_prompt=args.prompt,
        task_description=args.task,
        sample_inputs=sample_inputs,
        expected_outputs=expected_outputs,
        strategies=strategies
    )
    
    # Output results
    print("✅ Strategy comparison completed!")
    print("\n📊 Results:")
    
    for strategy, result in results.items():
        print(f"🧪 {strategy}: {result.performance_score:.2f}")
    
    if results:
        best = max(results.items(), key=lambda x: x[1].performance_score)
        print(f"\n🏆 Best strategy: {best[0]} (Score: {best[1].performance_score:.2f})")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(orchestrator.results_comparison, f, indent=2)
        print(f"💾 Results saved to {args.output}")

async def run_job_title_example(args):
    """Run job title classification example"""
    print("🎯 Running job title classification example...")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    example = JobTitleClassificationExample(api_key)
    
    if args.mode == 'quick':
        result = await example.run_basic_optimization()
        print(f"✅ Quick demo completed! Score: {result['result'].performance_score:.2f}")
    else:
        results = await example.run_complete_example()
        print("✅ Full demo completed!")
    
    if args.output:
        with open(args.output, 'w') as f:
            if args.mode == 'quick':
                json.dump({"result": result}, f, indent=2, default=str)
            else:
                json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output}")

async def run_evaluation(args):
    """Run prompt evaluation"""
    print("📊 Starting prompt evaluation...")
    
    # Load data
    inputs = YAMLParser.load_inputs_flexible(args.inputs)
    outputs = YAMLParser.load_inputs_flexible(args.outputs)
    
    expected_outputs = None
    if args.expected:
        expected_outputs = YAMLParser.load_expected_flexible(args.expected)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Setup evaluation
    evaluation_system = ComprehensiveEvaluationSystem(api_key)
    
    # Configure metrics
    metric_mapping = {
        'accuracy': EvaluationMetric.ACCURACY,
        'relevance': EvaluationMetric.RELEVANCE,
        'clarity': EvaluationMetric.CLARITY,
        'completeness': EvaluationMetric.COMPLETENESS,
        'consistency': EvaluationMetric.CONSISTENCY
    }
    
    criteria = []
    weight_per_metric = 1.0 / len(args.metrics)
    
    for metric_name in args.metrics:
        if metric_name in metric_mapping:
            criteria.append(EvaluationCriteria(
                metric=metric_mapping[metric_name],
                weight=weight_per_metric,
                description=f"Evaluation of {metric_name}"
            ))
    
    # Prepare evaluation data
    evaluations = []
    for i, (input_text, output) in enumerate(zip(inputs, outputs)):
        expected = expected_outputs[i] if expected_outputs and i < len(expected_outputs) else None
        evaluations.append({
            "prompt": args.prompt,
            "input": input_text,
            "output": output,
            "expected": expected
        })
    
    # Run evaluation
    results = await evaluation_system.batch_evaluate(evaluations, criteria)
    
    # Output results
    avg_score = sum(r.overall_score for r in results) / len(results)
    print(f"✅ Evaluation completed!")
    print(f"📈 Average score: {avg_score:.2f}")
    
    if args.output:
        evaluation_system.export_evaluation_results(results, args.output)
        print(f"💾 Results saved to {args.output}")

async def main():
    """Main CLI entry point"""
    parser = setup_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'optimize':
            await run_basic_optimization(args)
        elif args.command == 'compare':
            await run_strategy_comparison(args)
        elif args.command == 'job-title-example':
            await run_job_title_example(args)
        elif args.command == 'evaluate':
            await run_evaluation(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())