from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re
import ast

def validate_equation(equation_str, numbers, target):
    """
    Validates if the equation uses the correct numbers and equals the target.
    Returns (is_valid, error_message)
    """
    try:
        # Clean the equation string
        equation = equation_str.split('=')[0].strip()
        
        # Extract all numbers from the equation
        nums_in_equation = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are from the original list
        if not all(n in numbers for n in nums_in_equation):
            return False, "Equation uses numbers not in the original list"
        
        # Check if each number is used only once
        if len(nums_in_equation) != len(set(nums_in_equation)):
            return False, "Some numbers are used multiple times"
            
        # Evaluate the equation
        result = eval(equation)
        if abs(result - target) > 0.0001:  # Using small epsilon for float comparison
            return False, f"Equation equals {result}, not {target}"
            
        return True, "Valid equation"
        
    except Exception as e:
        return False, f"Error validating equation: {str(e)}"

def generate_response(numbers, target):
    """
    Generates a mathematical solution using the given numbers to reach the target.
    """
    # Example showing correct format
    example = """Example input:
    Numbers: [2, 4, 6, 8]
    Target: 14
    
    Example output:
    <think>
    1. Looking at numbers [2, 4, 6, 8] and target 14
    2. I can add 6 and 8 to get 14
    3. This is the simplest solution
    </think>
    <answer>6 + 8 = 14</answer>"""
    
    prompt = [{
        "role": "system",
        "content": f"""You are a math problem solver. Your task is to use the given numbers and basic arithmetic 
        operations (+, -, *, /) to reach the target number. Each number can only be used once.
        
        Rules:
        1. Use only the provided numbers
        2. Each number can be used exactly once
        3. Use only basic operations: +, -, *, /
        4. The final result must equal the target number
        
        {example}"""
    },
    {
        "role": "user", 
        "content": f"Numbers: {numbers}\nTarget: {target}\n\nShow your step-by-step thinking in <think></think> tags, then provide the final equation in <answer></answer> tags."
    },
    {
        "role": "assistant",
        "content": "I'll solve this step by step.\n<think>"
    }]
    
    # Initialize model and tokenizer if not already initialized
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device_map="mps",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, "qwen-r1-aha-moment")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")
    
    # Generate with more controlled parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,  # Reduced temperature for more focused outputs
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Validate response format and content
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    
    if not (think_match and answer_match):
        return "Error: Response missing required sections"
    
    # Extract equation from answer
    answer = answer_match.group(1).strip()
    is_valid, error_msg = validate_equation(answer, numbers, target)
    
    if not is_valid:
        return f"Error: Invalid solution - {error_msg}"
    
    return response

def main():
    try:
        numbers = [19, 36, 55, 7]
        target = 65
        print(f"\nSolving for numbers {numbers} to reach target {target}")
        print("-" * 50)
        
        response = generate_response(numbers, target)
        print("\nResponse:")
        print(response)
        
        # Verify format
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        print("\nParsed sections:")
        print(f"Thinking: {bool(think_match)}")
        print(f"Answer: {bool(answer_match)}")
        
        if think_match and answer_match:
            print("\nThinking section:")
            print(think_match.group(1).strip())
            print("\nAnswer section:")
            print(answer_match.group(1).strip())
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
