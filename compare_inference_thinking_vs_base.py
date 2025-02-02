from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import re
import time
from tqdm import tqdm

def load_model_and_tokenizer(model_id):
    """Load model and tokenizer from given model ID."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def create_prompt(sample):
    """Create conversation prompt from sample."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
    ]
    return messages

def generate_response(model, tokenizer, messages):
    """Generate response from model."""
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_answer(response):
    """Extract equation from response."""
    answer_match = re.search(r'<answer>(.*?)</answer>', response)
    return answer_match.group(1).strip() if answer_match else None

def evaluate_equation(equation, nums, target):
    """Check if equation is valid and equals target."""
    try:
        # Remove all whitespace from equation
        equation = ''.join(equation.split())
        
        # Check if all numbers are used exactly once
        used_nums = [str(n) for n in nums]
        eq_nums = re.findall(r'\d+', equation)
        if sorted(eq_nums) != sorted(used_nums):
            return False
        
        # Evaluate equation
        result = eval(equation)
        return abs(result - target) < 1e-10  # Account for floating point precision
    except:
        return False

def compare_models(num_samples=100, seed=42):
    """Compare fine-tuned model against base model."""
    random.seed(seed)
    
    # Load models
    fine_tuned_id = "philschmid/qwen-2.5-3b-r1-countdown"
    base_id = "Qwen/Qwen2.5-3B-Instruct"
    
    print("Loading models...")
    fine_tuned_model, fine_tuned_tokenizer = load_model_and_tokenizer(fine_tuned_id)
    base_model, base_tokenizer = load_model_and_tokenizer(base_id)
    
    # Load dataset
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    
    # Select random samples
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]
    
    results = {
        'fine_tuned': {'correct': 0, 'total': 0, 'time': 0},
        'base': {'correct': 0, 'total': 0, 'time': 0}
    }
    
    # Evaluate both models
    for sample in tqdm(samples, desc="Evaluating models"):
        messages = create_prompt(sample)
        
        # Evaluate fine-tuned model
        start_time = time.time()
        response = generate_response(fine_tuned_model, fine_tuned_tokenizer, messages)
        results['fine_tuned']['time'] += time.time() - start_time
        
        equation = extract_answer(response)
        if equation:
            results['fine_tuned']['total'] += 1
            if evaluate_equation(equation, sample['nums'], sample['target']):
                results['fine_tuned']['correct'] += 1
        
        # Evaluate base model
        start_time = time.time()
        response = generate_response(base_model, base_tokenizer, messages)
        results['base']['time'] += time.time() - start_time
        
        equation = extract_answer(response)
        if equation:
            results['base']['total'] += 1
            if evaluate_equation(equation, sample['nums'], sample['target']):
                results['base']['correct'] += 1
    
    # Calculate and print metrics
    print("\nResults:")
    for model_type, stats in results.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_time = stats['time'] / num_samples
        print(f"\n{model_type.replace('_', ' ').title()} Model:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Valid responses: {stats['total']}/{num_samples}")
        print(f"Average time per sample: {avg_time:.2f}s")

if __name__ == "__main__":
    compare_models(num_samples=100)  # Adjust number of samples as needed
