import torch
import re
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel  # Assuming you're still using Unsloth
from tqdm import tqdm  # For progress bars

# --- Configuration ---
MODEL_PATH = "path/to/your/trained/model"  # Replace with the actual path
# Or, if loading LoRA weights:
LORA_PATH = "path/to/your/saved/lora"  # Path to the saved LoRA weights
SYSTEM_PROMPT = """
You must respond in exactly this format, with no extra text before or after:

<thinking>
Show your step-by-step calculations using the provided numbers.
Each step must show a valid arithmetic operation (addition, subtraction, multiplication, or division).
You must use each number exactly once.
Each step should clearly show the operation and result.
</thinking>
<answer>
Your final calculated value here, as a single number.
</answer>
"""
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16  # Adjust based on your GPU memory
DATASET_NAME = "PengFeiChen/Countdown-Tasks-3to4"

# --- Helper Functions (from your training script) ---

def extract_xml_answer(text: str) -> str:
    """Extract content between <answer> tags"""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return ""

def extract_xml_thinking(text: str) -> str:
    """Extract content between <thinking> tags"""
    try:
        thinking = text.split("<thinking>")[-1]
        thinking = thinking.split("</thinking>")[0]
        return thinking.strip()
    except:
        return ""

def validate_arithmetic_step(step: str, available_nums: set) -> tuple[bool, float, set]:
    """
    Validate a single arithmetic step
    Returns: (is_valid, result, remaining_numbers)
    """
    try:
        # Extract numbers and operation
        numbers = [float(n) for n in re.findall(r'-?\d*\.?\d+', step)]
        ops = re.findall(r'[\+\-\*\/×÷]', step)

        if len(numbers) != 2 or len(ops) != 1:
            return False, None, available_nums

        # Validate numbers are available
        if not all(n in available_nums for n in numbers[:2]):
            return False, None, available_nums

        # Calculate result
        a, b = numbers[:2]
        op = ops[0]
        if op in ['+', 'plus']:
            result = a + b
        elif op in ['-', 'minus']:
            result = a - b
        elif op in ['*', '×', 'times']:
            result = a * b
        elif op in ['/', '÷']:
            if b == 0:
                return False, None, available_nums
            result = a / b
        else:
            return False, None, available_nums

        # Update available numbers
        remaining_nums = available_nums - {a, b}

        return True, result, remaining_nums
    except:
        return False, None, available_nums

def validate_arithmetic(thinking: str, nums: list) -> tuple[bool, float]:
    """
    Validate complete arithmetic solution
    Returns: (is_valid, final_result)
    """
    try:
        available_nums = set(map(float, nums))
        steps = [s.strip() for s in thinking.split('\n') if s.strip()]

        result = None
        for step in steps:
            is_valid, step_result, available_nums = validate_arithmetic_step(
                step, available_nums
            )
            if not is_valid:
                return False, None
            result = step_result

        # All numbers should be used
        if available_nums:
            return False, None

        return True, result
    except:
        return False, None

def evaluate_model(model, tokenizer, dataset, batch_size=BATCH_SIZE):
    """Evaluates the model on the given dataset."""

    results = []
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size

    model_for_inference = FastLanguageModel.for_inference(model)

    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total_samples)
        batch = dataset[batch_start:batch_end]

        prompts = []
        for example in batch:
            nums_str = ", ".join(map(str,example['nums']))
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Find a valid arithmetic result using these numbers exactly once: {nums_str}"}
                ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model_for_inference.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,  # Use a consistent temperature
                top_p=0.9,        # and top_p for generation
                do_sample=True,
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, response in enumerate(decoded_outputs):
            true_answer = batch['answer'][j]
            nums = batch['nums'][j]
            thinking = extract_xml_thinking(response)
            model_answer = extract_xml_answer(response)
            is_valid, calculated_result = validate_arithmetic(thinking, nums)
            
            try:
                answer_correct = float(model_answer) == float(true_answer)
            except (ValueError, TypeError):
                answer_correct = False

            results.append({
                "question": prompts[j],
                "true_answer": true_answer,
                "model_answer": model_answer,
                "thinking": thinking,
                "is_arithmetically_valid": is_valid,
                "answer_is_correct": answer_correct,
                "full_response": response,
            })

    return results

def calculate_metrics(results):
    """Calculates and returns evaluation metrics."""

    total = len(results)
    valid_count = sum(1 for r in results if r['is_arithmetically_valid'])
    correct_count = sum(1 for r in results if r['answer_is_correct'])
    valid_and_correct_count = sum(1 for r in results if r['is_arithmetically_valid'] and r['answer_is_correct'])

    validity_rate = (valid_count / total) * 100 if total else 0.0
    accuracy = (correct_count / total) * 100 if total else 0.0
    valid_accuracy = (valid_and_correct_count / valid_count) * 100 if valid_count else 0.0
    
    # Check Format
    pattern = r"^<thinking>.*?</thinking>\s*<answer>.*?</answer>\s*$"
    format_count = sum([1 for r in results if bool(re.match(pattern, r['full_response'], re.DOTALL))])
    format_rate = (format_count / total) * 100 if total else 0.0

    return {
        "total_examples": total,
        "arithmetic_validity_rate (%)": validity_rate,
        "overall_accuracy (%)": accuracy,
        "valid_accuracy (%)": valid_accuracy,  # Accuracy *given* valid reasoning
        "format_correctness (%)": format_rate
    }

def main():
    """Main evaluation function."""

    # Load model and tokenizer
    if LORA_PATH:
        # Load base model (you might need to adjust parameters)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Phi-4",  # Or the base model you used
            max_seq_length=256,  # Use appropriate max_seq_length
            load_in_4bit=True, # Or 8bit, or neither
            torch_dtype=torch.bfloat16, # use torch.float16 if not using bfloat16
            device_map="auto"
        )
        model.load_lora(LORA_PATH)  # Load the LoRA weights

    else:  # Load a full fine-tuned model
      model, tokenizer = FastLanguageModel.from_pretrained(
          model_name=MODEL_PATH,
          max_seq_length=256,  # Use appropriate max_seq_length
            load_in_4bit=True, # Or 8bit, or neither
            torch_dtype=torch.bfloat16, # use torch.float16 if not using bfloat16
          device_map="auto"
      )
    model.eval()  # Set the model to evaluation mode

    # Load the test dataset (using your improved loading function)
    test_dataset = load_dataset(DATASET_NAME, split="test")

    # Evaluate the model
    eval_results = evaluate_model(model, tokenizer, test_dataset)

    # Calculate metrics
    metrics = calculate_metrics(eval_results)

    # Print results
    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    # Save results (optional, but recommended)
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"\nDetailed results saved to: {results_file}")

    # --- Manual Inspection (Highly Recommended) ---
    print("\n--- Manual Inspection (First 10 Examples) ---")
    for i in range(min(10, len(eval_results))):  # Inspect the first 10 examples
        result = eval_results[i]
        print(f"\nExample {i + 1}:")
        print(f"  Question: {result['question']}")
        print(f"  True Answer: {result['true_answer']}")
        print(f"  Model Answer: {result['model_answer']}")
        print(f"  Thinking: {result['thinking']}")
        print(f"  Arithmetically Valid: {result['is_arithmetically_valid']}")
        print(f"  Answer Correct: {result['answer_is_correct']}")
        print(f"  Full Response: {result['full_response']}")


if __name__ == "__main__":
    main()
