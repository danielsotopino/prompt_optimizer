"""
Lambda Functions for Prompt Optimization
Based on the MetaGPT notebook approach using lambda functions for dynamic prompt generation
"""
from typing import Dict, Any, Callable, List
import json

class PromptLambdaFactory:
    """Factory for creating lambda functions for prompt optimization as shown in MetaGPT notebook"""
    
    @staticmethod
    def create_job_classification_lambda() -> Callable[[str], str]:
        """
        Creates lambda function for job title classification as shown in the notebook
        """
        return lambda job_title: f"""Classify the following job title into appropriate categories and subcategories.

Consider these guidelines:
- Technology roles should be categorized by specific domain (Software Engineering, Data Science, DevOps, etc.)
- Business roles should include functional area (Sales, Marketing, Operations, etc.)
- Executive roles should specify the department or function
- Handle ambiguous titles by providing the most likely classification
- Use clear, industry-standard categorization

Format your response as: "Category - Subcategory"

Job Title: {job_title}

Classification:"""
    
    @staticmethod
    def create_enhanced_job_classification_lambda() -> Callable[[str], str]:
        """
        Enhanced version with examples and edge case handling (optimized version from notebook)
        """
        return lambda job_title: f"""You are an expert HR professional specializing in job classification. Classify the following job title into appropriate categories and subcategories.

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
    
    @staticmethod
    def create_generic_optimization_lambda(task_description: str, examples: List[str] = None) -> Callable[[str], str]:
        """
        Generic lambda function for any optimization task
        """
        examples_text = ""
        if examples:
            examples_text = "\n\nExamples:\n" + "\n".join(f"- {example}" for example in examples)
        
        return lambda input_text: f"""Task: {task_description}

{examples_text}

Input: {input_text}

Response:"""
    
    @staticmethod
    def create_few_shot_lambda(task_description: str, examples: List[Dict[str, str]]) -> Callable[[str], str]:
        """
        Creates few-shot prompting lambda as demonstrated in the notebook
        """
        few_shot_examples = ""
        for i, example in enumerate(examples, 1):
            few_shot_examples += f"\nExample {i}:\nInput: {example['input']}\nOutput: {example['output']}\n"
        
        return lambda input_text: f"""Task: {task_description}

{few_shot_examples}

Now apply the same pattern:
Input: {input_text}
Output:"""
    
    @staticmethod
    def create_structured_output_lambda(task_description: str, output_schema: Dict[str, str]) -> Callable[[str], str]:
        """
        Creates lambda for structured output generation
        """
        schema_text = json.dumps(output_schema, indent=2)
        
        return lambda input_text: f"""Task: {task_description}

Required Output Schema:
{schema_text}

Provide your response in the exact JSON format specified above.

Input: {input_text}

Response:"""

class MetaGPTStyleOptimizer:
    """
    Optimizer that follows the exact MetaGPT notebook methodology
    """
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.optimization_rounds = []
    
    async def optimize_with_lambda(
        self, 
        initial_lambda: Callable[[str], str],
        task_description: str,
        test_inputs: List[str],
        expected_outputs: List[str] = None,
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize prompt using lambda function approach from MetaGPT notebook
        """
        current_lambda = initial_lambda
        best_score = 0.0
        best_lambda = initial_lambda
        
        for round_num in range(1, max_rounds + 1):
            print(f"🔄 Optimization Round {round_num}")
            
            # Test current lambda
            current_score = await self._evaluate_lambda(current_lambda, test_inputs, expected_outputs)
            
            # Store round results
            round_result = {
                "round": round_num,
                "score": current_score,
                "prompt_template": self._extract_prompt_from_lambda(current_lambda),
                "improvements": []
            }
            
            if current_score > best_score:
                best_score = current_score
                best_lambda = current_lambda
            
            # Generate improved lambda for next round
            if round_num < max_rounds:
                improved_lambda = await self._generate_improved_lambda(
                    current_lambda, task_description, test_inputs, round_result
                )
                current_lambda = improved_lambda
            
            self.optimization_rounds.append(round_result)
            print(f"📈 Round {round_num} Score: {current_score:.2f}")
        
        return {
            "best_lambda": best_lambda,
            "best_score": best_score,
            "optimization_history": self.optimization_rounds,
            "final_prompt_template": self._extract_prompt_from_lambda(best_lambda)
        }
    
    async def _evaluate_lambda(
        self, 
        lambda_func: Callable[[str], str], 
        test_inputs: List[str], 
        expected_outputs: List[str] = None
    ) -> float:
        """Evaluate lambda function performance"""
        total_score = 0.0
        
        for i, test_input in enumerate(test_inputs):
            try:
                # Generate prompt using lambda
                prompt = lambda_func(test_input)
                
                # Execute with LLM
                response = self.client.chat_completions_create(
                    model=self.config.execution_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                output = response.choices[0].message.content.strip()
                
                # Score the output
                if expected_outputs and i < len(expected_outputs):
                    score = await self._score_against_expected(output, expected_outputs[i])
                else:
                    score = await self._score_quality(output, test_input)
                
                total_score += score
                
            except Exception as e:
                print(f"⚠️ Error evaluating input {i+1}: {str(e)}")
                continue
        
        return total_score / len(test_inputs) if test_inputs else 0.0
    
    async def _generate_improved_lambda(
        self,
        current_lambda: Callable[[str], str],
        task_description: str,
        test_inputs: List[str],
        round_result: Dict[str, Any]
    ) -> Callable[[str], str]:
        """Generate improved lambda function"""
        
        current_prompt_template = self._extract_prompt_from_lambda(current_lambda)
        
        improvement_prompt = f"""You are an expert prompt engineer. Analyze and improve this prompt template.

Current Task: {task_description}
Current Score: {round_result['score']:.2f}

Current Prompt Template:
```
{current_prompt_template}
```

Test Cases:
{chr(10).join(f"- {inp}" for inp in test_inputs[:3])}

Improve this prompt by:
1. Adding more specific instructions
2. Including relevant examples if beneficial
3. Improving clarity and structure
4. Handling edge cases better
5. Optimizing for the specific task

Return ONLY the improved prompt template with {{input_variable}} as placeholder."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.optimization_model,
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            improved_template = response.choices[0].message.content.strip()
            
            # Create new lambda function
            return lambda input_text: improved_template.replace("{input_variable}", input_text)
            
        except Exception as e:
            print(f"⚠️ Error generating improvement: {str(e)}")
            return current_lambda
    
    def _extract_prompt_from_lambda(self, lambda_func: Callable[[str], str]) -> str:
        """Extract prompt template from lambda function"""
        try:
            # Test with placeholder to extract template
            sample_output = lambda_func("{input_variable}")
            return sample_output
        except:
            return "Unable to extract prompt template"
    
    async def _score_against_expected(self, output: str, expected: str) -> float:
        """Score output against expected result"""
        evaluation_prompt = f"""Rate the similarity between these two outputs on a scale of 0.0 to 1.0:

Expected: {expected}
Actual: {output}

Consider semantic similarity, not just exact matching. Return only the numerical score."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.evaluation_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except:
            return 0.5
    
    async def _score_quality(self, output: str, input_text: str) -> float:
        """Score output quality when no expected output available"""
        evaluation_prompt = f"""Rate the quality of this response on a scale of 0.0 to 1.0:

Input: {input_text}
Output: {output}

Consider relevance, completeness, and accuracy. Return only the numerical score."""

        try:
            response = self.client.chat_completions_create(
                model=self.config.evaluation_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except:
            return 0.5

# Example usage functions matching the notebook style
def create_job_title_classifier():
    """Create job title classifier using lambda approach"""
    return PromptLambdaFactory.create_job_classification_lambda()

def create_enhanced_job_title_classifier():
    """Create enhanced job title classifier"""
    return PromptLambdaFactory.create_enhanced_job_classification_lambda()

def optimize_job_title_classification(client, config):
    """Full optimization example as shown in MetaGPT notebook"""
    
    # Sample data from notebook
    test_job_titles = [
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
    
    expected_classifications = [
        "Technology - Software Engineering",
        "Technology - Data Science",
        "Business - Marketing", 
        "Executive - Technology",
        "Technology - Software Engineering",
        "Technology - AI/ML",
        "Technology - DevOps",
        "Creative - Design",
        "Business - Sales",
        "Business - Finance"
    ]
    
    # Initial lambda (basic version)
    initial_lambda = PromptLambdaFactory.create_job_classification_lambda()
    
    # Optimizer
    optimizer = MetaGPTStyleOptimizer(client, config)
    
    return optimizer.optimize_with_lambda(
        initial_lambda=initial_lambda,
        task_description="Classify job titles into appropriate categories and subcategories",
        test_inputs=test_job_titles,
        expected_outputs=expected_classifications,
        max_rounds=5
    )