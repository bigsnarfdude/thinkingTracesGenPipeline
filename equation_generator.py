from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    system_message: str = "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    user_template: str = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
    assistant_prefix: str = "Let me solve this step by step.\n<think>"

class EquationPromptGenerator:
    """Generator for equation-solving prompts"""
    
    def __init__(self, model_name: str, prompt_config: Optional[PromptConfig] = None):
        """
        Initialize the prompt generator
        
        Args:
            model_name: Name of the model/tokenizer to use
            prompt_config: Configuration for prompt generation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = prompt_config or PromptConfig()
    
    def generate_prompt(self, numbers: Union[List[int], List[float]], target: Union[int, float]) -> Dict[str, Any]:
        """
        Generate a prompt for equation creation
        
        Args:
            numbers: List of numbers to use in equation
            target: Target value the equation should equal
            
        Returns:
            Dictionary containing prompt and target value
        """
        messages = [
            {
                "role": "system",
                "content": self.config.system_message
            },
            {
                "role": "user",
                "content": self.config.user_template.format(
                    numbers=numbers,
                    target=target
                )
            },
            {
                "role": "assistant",
                "content": self.config.assistant_prefix
            }
        ]
        
        return {
            "prompt": self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True
            ),
            "target": target
        }

    def batch_generate(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate prompts for multiple problems
        
        Args:
            problems: List of dictionaries, each containing 'numbers' and 'target'
            
        Returns:
            List of dictionaries containing prompts and targets
        """
        return [
            self.generate_prompt(
                problem['numbers'],
                problem['target']
            )
            for problem in problems
        ]

# Example usage:
def example_usage():
    # Initialize generator with custom config
    custom_config = PromptConfig(
        system_message="You are a math tutor...",
        user_template="Find an equation using {numbers} that equals {target}...",
        assistant_prefix="I'll help you solve this..."
    )
    
    generator = EquationPromptGenerator(
        model_name="your-model-name",
        prompt_config=custom_config
    )
    
    # Generate single prompt
    result = generator.generate_prompt([1, 2, 3, 4], 10)
    
    # Generate multiple prompts
    problems = [
        {"numbers": [1, 2, 3], "target": 6},
        {"numbers": [2, 4, 6], "target": 12}
    ]
    batch_results = generator.batch_generate(problems)
    
    return result, batch_results

if __name__ == "__main__":
    # Example of how to use the module
    single_result, batch_results = example_usage()
    print("Single Result:", single_result)
    print("Batch Results:", batch_results)
