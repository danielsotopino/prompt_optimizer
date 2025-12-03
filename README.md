# Self-Supervised Prompt Optimization (SPO) Framework

A complete implementation of the SPO framework based on MetaGPT for automated prompt optimization using OpenRouter. This project supports multiple LLM providers (OpenAI, Anthropic, Google, etc.) through OpenRouter's unified API.

## 🚀 Features

- **Iterative Optimization**: Automatic prompt improvement through multiple iterations
- **Multiple Strategies**: Support for different optimization approaches
- **Comprehensive Evaluation System**: Comprehensive metrics to evaluate prompt quality
- **Advanced Pipeline**: Strategy comparison and multi-objective optimization
- **MetaGPT Lambda Functions**: Faithful implementation of the original notebook with lambda functions
- **Confidence Scoring**: Advanced system to measure result reliability
- **Automatic Stopping Criteria**: Intelligent convergence detection
- **YAML/JSON Support**: Flexible formats for defining prompts and inputs
- **Practical Example**: Complete demonstration with job title classification

## 📁 Project Structure

```
prompt_optimizer/
├── spo_framework.py          # Main SPO framework
├── optimization_pipeline.py  # Advanced pipeline with multiple strategies
├── evaluation_system.py      # Comprehensive evaluation system
├── lambda_functions.py       # MetaGPT-style lambda functions
├── confidence_scoring.py     # Confidence scoring system
├── yaml_parser.py            # Parser for YAML/JSON files
├── main.py                   # CLI interface
├── requirements.txt          # Dependencies (includes PyYAML)
├── .env                      # Environment variables (API keys)
├── .gitignore                # Files to ignore in git
├── examples/                 # Configuration examples (✅ Tested)
│   ├── sample_inputs.json        # Basic example in JSON
│   ├── sample_inputs.yaml        # Basic example in YAML
│   ├── sample_expected.json      # Expected outputs for examples
│   ├── classification_example.yaml # Complete classification example
│   └── prompt_config_example.yaml  # Configuration with included prompt
└── README.md                # This file
```

## 🛠️ Installation

1. **Create Python 3.11 virtual environment**:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure OpenRouter API Key**:
```bash
# Create .env file or export environment variable
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Configure models in .env file (see .env.example)
# OPTIMIZATION_MODEL=openai/gpt-4o
# EXECUTION_MODEL=openai/gpt-4o-mini
# EVALUATION_MODEL=openai/gpt-4o
```

## 🎯 Quick Start

### 🚀 Quick Framework Test (Recommended to start)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Configure OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# 3. Run quick test (2-3 minutes, ~$0.10-0.20)
python test_optimization.py
```

**What the quick test does:**
- ⚡ 3 basic optimization iterations
- 🎯 Use case: job title classification
- 📊 Shows scores and improvements in real-time
- 💰 Uses `openai/gpt-4o-mini` for lower cost
- ⏱️ Completes in 2-3 minutes

### Job Title Classification Example

```bash
# Quick demo (5-10 minutes, ~$0.50-1.00)
python main.py job-title-example --mode quick --output quick_results.json

# Full demo with all features (15-20 minutes, ~$2.00-4.00)
python main.py job-title-example --mode full --output full_results.json
```

### 🎛️ Advanced Testing Options

#### Custom Optimization

```bash
# Using traditional JSON files
python main.py optimize \
  --prompt "Classify the job title into categories. Format: Category - Subcategory. Title: {job_title}" \
  --task "Job title classification" \
  --inputs examples/sample_inputs.json \
  --expected examples/sample_expected.json \
  --iterations 5 \
  --output optimization_results.json

# Using YAML files (new)
python main.py optimize \
  --prompt "Classify the job title into categories" \
  --task "Job title classification" \
  --inputs examples/sample_inputs.yaml \
  --expected examples/sample_expected.yaml \
  --iterations 5 \
  --output optimization_results.json

# Using complete YAML file with included prompt
python main.py optimize \
  --inputs examples/classification_example.yaml \
  --output optimization_results.json
```

#### Strategy Comparison
```bash
# With JSON files
python main.py compare \
  --prompt "Classify the job title into appropriate categories" \
  --task "Job title classification" \
  --inputs examples/sample_inputs.json \
  --strategies iterative_refinement ensemble_voting multi_objective \
  --output strategy_comparison.json

# With YAML files
python main.py compare \
  --inputs examples/classification_example.yaml \
  --strategies iterative_refinement ensemble_voting \
  --output strategy_comparison.json
```

#### Evaluation Only (no optimization)
```bash
# With JSON files
python main.py evaluate \
  --prompt "Your prompt to evaluate" \
  --inputs examples/sample_inputs.json \
  --outputs examples/sample_outputs.json \
  --expected examples/sample_expected.json \
  --metrics accuracy relevance clarity \
  --output evaluation_results.json

# With YAML files
python main.py evaluate \
  --prompt "Your prompt to evaluate" \
  --inputs examples/classification_example.yaml \
  --outputs examples/outputs.yaml \
  --expected examples/classification_example.yaml \
  --metrics accuracy relevance clarity \
  --output evaluation_results.json
```

## 📝 YAML and JSON Format Support

### Complete YAML Format (Recommended)

The framework now supports YAML files that include the prompt, input data, and expected output all in one file:

```yaml
prompt: |
  You are an agent specialized in classifying user messages.
  Your task is to analyze each message and classify it into one of the following categories:
  - business_relevant: Messages related to business or product inquiries
  - support_request: Help requests or technical support
  - complaint: Complaints or expressions of dissatisfaction
  - compliment: Praise or positive comments
  
  Respond only with the corresponding category.

task: "Classify user messages into predefined categories"

data:
  - message: "Hello, do you have computers in stock?"
    classification: "business_relevant"
  - message: "My account is not working, I need help"
    classification: "support_request"
  - message: "I'm very upset because my order arrived late"
    classification: "complaint"
  - message: "Excellent service, highly recommended"
    classification: "compliment"

iterations: 5
model: "gpt-4o"
```

### Supported Formats

1. **YAML with combined data** (like the example above)
2. **YAML with separate sections**:
```yaml
inputs:
  - "Message 1"
  - "Message 2"
expected:
  - "Category 1"
  - "Category 2"
```

3. **Traditional JSON** (maintains compatibility):
```json
["input1", "input2", "input3"]
```

### YAML Parser Usage

```python
from yaml_parser import YAMLParser

# Load inputs from any format
inputs = YAMLParser.load_inputs_flexible("examples/file.yaml")

# Load expected outputs
expected = YAMLParser.load_expected_flexible("examples/file.yaml")

# Load complete configuration
config = YAMLParser.parse_prompt_config("examples/classification_example.yaml")
```

## 📂 Examples Directory

The `examples/` directory contains example files for different use cases:

### 📄 Available Files

1. **`sample_inputs.json`** - Basic example in JSON format
   ```json
   [
     "Senior Software Engineer at Google",
     "Data Scientist - Machine Learning",
     "Marketing Manager, Digital Products"
   ]
   ```

2. **`sample_inputs.yaml`** - Same content in YAML format
   ```yaml
   inputs:
     - "Senior Software Engineer at Google"
     - "Data Scientist - Machine Learning"
     - "Marketing Manager, Digital Products"
   ```

3. **`classification_example.yaml`** - Complete example with prompt and data
   ```yaml
   prompt: |
     Classify messages into specific categories...
   
   data:
     - message: "Hello, do you have stock?"
       classification: "business_relevant"
   ```

4. **`prompt_config_example.yaml`** - Configuration with multiple parameters
   ```yaml
   prompt: |
     Your prompt here...
   task: "Task description"
   iterations: 5
   model: "gpt-4o"
   strategies: ["iterative_refinement"]
   ```

### 🚀 How to Use the Examples (✅ All Tested)

```bash
# 1. Basic example with external prompt
python main.py optimize \
  --inputs examples/sample_inputs.yaml \
  --prompt "Classify job titles into categories" \
  --task "Job title classification" \
  --iterations 2 \
  --output results.yaml

# 2. Complete example with included prompt
python main.py optimize \
  --inputs examples/classification_example.yaml \
  --iterations 3 \
  --output classification_results.yaml

# 3. Using traditional JSON files
python main.py optimize \
  --prompt "Classify these job titles" \
  --task "Job classification" \
  --inputs examples/sample_inputs.json \
  --expected examples/sample_expected.json \
  --output traditional_results.json
```

### ✅ Test Results

The examples have been tested and work correctly:

- **`classification_example.yaml`**: ✅ Perfect score (1.00) in 1 iteration
- **`prompt_config_example.yaml`**: ✅ Perfect score (1.00) in 1 iteration
- **`sample_inputs.yaml`**: ✅ Improved score (0.68) in 2 iterations  
- **YAML output**: ✅ Readable and well-structured format
- **All formats**: ✅ JSON and YAML working correctly

### YAML Format Advantages

- **Multiline prompts**: Use `|` for long and complex prompts
- **Single file**: Everything in one place (prompt, data, configuration)
- **More readable**: Clearer format than JSON
- **Comments**: You can add comments with `#`
- **Flexible**: Multiple formats supported
- **YAML output**: Results can also be saved in YAML

### 💾 Output Formats

The framework automatically detects the output format based on the file extension:

```bash
# JSON output (traditional)
python main.py optimize --inputs examples/classification_example.yaml --output results.json

# YAML output (new, more readable)
python main.py optimize --inputs examples/classification_example.yaml --output results.yaml
```

**YAML output example:**
```yaml
config:
  max_iterations: 5
  optimization_model: "gpt-4o"
  execution_model: "gpt-4o-mini"

optimization_history:
  - iteration: 1
    performance_score: 0.91
    optimized_prompt: "Optimized prompt..."
    execution_time: 45.2

summary:
  total_iterations: 5
  best_score: 0.95
  improvement: 0.15
  best_prompt: "Best prompt found..."
```

## 📊 Optimization Strategies

### 1. Iterative Refinement (`iterative_refinement`)
- **Best for**: General-purpose optimization
- **Description**: Step-by-step incremental improvement
- **Advantages**: Fast and efficient

### 2. Ensemble Voting (`ensemble_voting`)
- **Best for**: Robust and reliable results
- **Description**: Multiple optimizations in parallel
- **Advantages**: Greater stability and accuracy

### 3. Multi-Objective (`multi_objective`)
- **Best for**: Balancing competing objectives
- **Description**: Optimizes for multiple criteria simultaneously
- **Advantages**: Balanced solutions

### 4. Genetic Algorithm (`genetic_algorithm`)
- **Best for**: Creative and diverse exploration
- **Description**: Evolution of prompts through generations
- **Advantages**: Finds innovative solutions

## 🔧 Programmatic Usage

### Basic Optimization

```python
import asyncio
from spo_framework import SPOFramework, PromptOptimizationConfig

async def optimize_prompt():
    config = PromptOptimizationConfig(
        max_iterations=5,
        optimization_model="gpt-4o",
        execution_model="gpt-4o-mini"
    )
    
    framework = SPOFramework(config, "your-api-key")
    
    result = await framework.optimize_prompt(
        initial_prompt="Your initial prompt",
        task_description="Task description",
        sample_inputs=["input1", "input2", "input3"],
        expected_outputs=["output1", "output2", "output3"]
    )
    
    print(f"Score: {result.performance_score}")
    print(f"Optimized prompt: {result.optimized_prompt}")

asyncio.run(optimize_prompt())
```

### MetaGPT-Style Optimization with Lambda Functions

```python
from lambda_functions import MetaGPTStyleOptimizer, PromptLambdaFactory

async def metagpt_optimization():
    # Create lambda function for job classification
    job_classifier = PromptLambdaFactory.create_job_classification_lambda()
    
    # MetaGPT-style optimizer
    optimizer = MetaGPTStyleOptimizer(client, config)
    
    result = await optimizer.optimize_with_lambda(
        initial_lambda=job_classifier,
        task_description="Classify job titles",
        test_inputs=job_titles,
        expected_outputs=expected_classifications,
        max_rounds=5
    )
```

### Advanced Pipeline

```python
from optimization_pipeline import AdvancedOptimizationPipeline, PipelineConfig, OptimizationStrategy

async def advanced_optimization():
    pipeline_config = PipelineConfig(
        strategy=OptimizationStrategy.ENSEMBLE_VOTING,
        ensemble_size=5
    )
    
    pipeline = AdvancedOptimizationPipeline(spo_config, pipeline_config, api_key)
    
    result = await pipeline.optimize_with_strategy(
        initial_prompt="Your prompt",
        task_description="Task",
        sample_inputs=inputs,
        expected_outputs=outputs
    )
```

### Confidence System

```python
from confidence_scoring import ConfidenceAnalyzer

async def analyze_confidence():
    analyzer = ConfidenceAnalyzer(client)
    
    confidence_metrics = await analyzer.analyze_optimization_confidence(
        optimization_history=optimization_results,
        test_inputs=test_data,
        final_prompt=best_prompt,
        num_validation_runs=5
    )
    
    report = analyzer.generate_confidence_report(confidence_metrics)
    print(f"Confidence Level: {report['confidence_level']}")
    print(f"Reliability Index: {report['reliability_index']:.2f}")
```

### Comprehensive Evaluation

```python
from evaluation_system import ComprehensiveEvaluationSystem, EvaluationCriteria, EvaluationMetric

async def evaluate_prompt():
    evaluation_system = ComprehensiveEvaluationSystem(api_key)
    
    criteria = [
        EvaluationCriteria(EvaluationMetric.ACCURACY, 0.5, "Accuracy"),
        EvaluationCriteria(EvaluationMetric.CLARITY, 0.3, "Clarity"),
        EvaluationCriteria(EvaluationMetric.RELEVANCE, 0.2, "Relevance")
    ]
    
    result = await evaluation_system.comprehensive_evaluate(
        prompt="Your prompt",
        input_text="Test input",
        output="Generated output",
        expected="Expected output",
        evaluation_criteria=criteria
    )
```

## 📈 Evaluation Metrics

- **Accuracy**: Precision and correctness of output
- **Relevance**: Relevance to input and task
- **Clarity**: Clarity and readability
- **Completeness**: Completeness of response
- **Consistency**: Consistency across multiple executions
- **Efficiency**: Prompt efficiency
- **Adherence to Format**: Adherence to specified format
- **Confidence Metrics**: Reliability and stability metrics

## 🔍 Monitoring and Results

Results can be exported in JSON format for later analysis:

```json
{
  "config": {...},
  "optimization_history": [...],
  "summary": {
    "total_iterations": 5,
    "best_score": 0.89,
    "improvement": 0.34,
    "best_prompt": "Final optimized prompt"
  }
}
```

## 🎯 Detailed Example: Job Title Classification

The framework includes a complete example that demonstrates optimization of a job title classification system:

```python
from job_title_example import JobTitleClassificationExample

async def run_example():
    example = JobTitleClassificationExample(api_key)
    results = await example.run_complete_example()
    
    # Runs:
    # 1. Basic SPO optimization
    # 2. Strategy comparison
    # 3. Comprehensive evaluation
    # 4. Final report with recommendations
```

## 🚀 Use Cases

- **Chatbot Improvement**: Optimize prompts for more natural conversations
- **Text Classification**: Improve accuracy in categorization tasks
- **Content Generation**: Optimize prompts for quality content
- **Sentiment Analysis**: Fine-tune prompts for better emotional detection
- **Information Extraction**: Improve prompts for extracting structured data

## 🔧 Advanced Configuration

### Custom Models

Models are configured via environment variables in `.env` file:

```bash
# .env file
OPENROUTER_API_KEY=your-api-key
OPTIMIZATION_MODEL=openai/gpt-4o
EXECUTION_MODEL=openai/gpt-4o-mini
EVALUATION_MODEL=openai/gpt-4o
```

Or via code:

```python
config = PromptOptimizationConfig(
    optimization_model="anthropic/claude-3.5-sonnet",  # Model for optimization
    execution_model="anthropic/claude-3-haiku",        # Model for execution
    evaluation_model="anthropic/claude-3.5-sonnet",    # Model for evaluation
    temperature=0.7,                                    # Creativity
    max_tokens=2000                                     # Token limit
)
```

### Custom Evaluation Criteria

```python
custom_criteria = [
    EvaluationCriteria(EvaluationMetric.ACCURACY, 0.4, "Technical precision"),
    EvaluationCriteria(EvaluationMetric.CREATIVITY, 0.3, "Creativity"),
    EvaluationCriteria(EvaluationMetric.SAFETY, 0.3, "Content safety")
]
```

## 📚 Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Original MetaGPT Paper](https://arxiv.org/abs/2308.00352)
- [OpenRouter Models](https://openrouter.ai/models)
- [Mistral Prompt Optimization - Base Document](https://docs.mistral.ai/guides/prompting_capabilities/) - *Optimization techniques that inspired this framework*

## 🤝 Contributions

Contributions are welcome. Please:

1. Fork the repository
2. Create a branch for your feature
3. Commit your changes
4. Open a Pull Request

## 📄 License

This project is under the MIT license. See LICENSE file for details.

## ⚠️ Important Considerations

### 💰 **Estimated API Costs**
- **Quick Test**: $0.10-0.20 (recommended to start)
- **Quick Demo**: $0.50-1.00
- **Full Demo**: $2.00-4.00
- **Custom Optimization**: $1.00-3.00 (depends on iterations)

### ⏱️ **Execution Times**
- **Quick Test**: 2-3 minutes
- **Quick Demo**: 5-10 minutes  
- **Full Demo**: 15-20 minutes
- **Strategy Comparison**: 10-15 minutes

### 🎯 **Tips for Better Results**
- **Quality Data**: Results depend on the quality of input data
- **Configuration**: Adjust parameters according to your specific needs
- **API Key**: Make sure you have sufficient credits in your OpenRouter account
- **First Tests**: Use `test_optimization.py` to validate your configuration

## 🚨 Common Troubleshooting

### ❌ "OPENAI_API_KEY NOT configured"
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### ❌ "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### ❌ "OpenRouter authentication error"
- Verify that your API key is correct
- Make sure you have credits in your OpenRouter account
- Check that there are no extra spaces in the API key
- Verify the model names are in the correct format (e.g., "openai/gpt-4o")

### ❌ "Timeout or network errors"
- Reduce the number of iterations: `--iterations 2`
- Use smaller models in configuration
- Verify your internet connection

## 🆘 Support

For problems or questions:
1. Review the documentation
2. Run `python test_optimization.py` for diagnosis
3. Try the examples in `examples/` directory
4. Check error logs in the console
5. Search in existing issues
6. Create a new issue with specific details

## 🎉 What's New in This Version

### ✨ Complete YAML Support

- **Flexible input**: YAML files with integrated prompts, tasks, and data
- **YAML output**: More readable and structured results  
- **Automatic detection**: System detects format by file extension
- **Full compatibility**: Maintains complete JSON support

### 📂 Organized Examples Directory

- **5 tested examples**: From basic to advanced
- **Real use cases**: Job classification, message classification
- **Multiple formats**: JSON and YAML demonstrated
- **Verified results**: All examples work correctly

### 🔧 Interface Improvements

- **More flexible CLI**: Optional arguments when in YAML
- **Better organization**: Files separated by function
- **Robust parser**: Error handling and multiple formats
- **Environment configuration**: Automatic loading of environment variables

## 🎯 Recommended Steps to Get Started

1. **Initial configuration:**
   ```bash
   source venv/bin/activate
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   # Or create .env file with:
   # OPENROUTER_API_KEY=your-key
   # OPTIMIZATION_MODEL=openai/gpt-4o
   # EXECUTION_MODEL=openai/gpt-4o-mini
   # EVALUATION_MODEL=openai/gpt-4o
   ```

2. **First test:**
   ```bash
   python test_optimization.py
   ```

3. **If it works well, try the demo:**
   ```bash
   python main.py job-title-example --mode quick
   ```

4. **Explore advanced features according to your needs**

---

**Start optimizing your prompts today!** 🚀