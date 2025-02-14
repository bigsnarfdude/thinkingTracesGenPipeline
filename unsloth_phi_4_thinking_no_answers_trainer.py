import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)

import torch
import re
import json
from datetime import datetime
from pathlib import Path
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Create logs directory
log_dir = Path("training_logs")
log_dir.mkdir(exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = log_dir / f"run_{current_time}"
run_dir.mkdir(exist_ok=True)

def log_interaction(question, true_answer, response, step):
    """Log each training interaction to a JSON file"""
    try:
        thinking = ""
        if "<thinking>" in response and "</thinking>" in response:
            thinking = response.split("<thinking>")[1].split("</thinking>")[0].strip()

        model_answer = ""
        if "<answer>" in response and "</answer>" in response:
            model_answer = response.split("<answer>")[1].split("</answer>")[0].strip()

        log_entry = {
            "step": step,
            "question": question,
            "true_answer": true_answer,
            "thinking": thinking,
            "model_answer": model_answer,
            "full_response": response
        }

        log_file = run_dir / "interactions.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"Error logging interaction: {str(e)}")

# Configuration
max_seq_length = 256
lora_rank = 8
gpu_memory_utilization = 0.7

print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model with memory optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gpu_memory_utilization,
)

# Load existing LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load the GSM8K trained LoRA weights
gsm8k_lora_path = "/home/vincent/dev/thinking/training_logs/run_20250209_061540/grpo_saved_lora"
model.load_lora(gsm8k_lora_path)

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

def get_countdown_questions(split="train") -> Dataset:
    """Load and format dataset without revealing target answers"""
    data = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Find a valid arithmetic result using these numbers exactly once: {x['nums']}"}
        ],
        'answer': str(x['target']),  # Kept for validation but not shown to model
        'nums': x['nums']  # Store original numbers for validation
    })
    return data

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

# Reward functions
def arithmetic_reward_func(completions, nums, **kwargs) -> list[float]:
    """Reward for valid arithmetic operations"""
    rewards = []
    for completion in completions:
        thinking = extract_xml_thinking(completion[0]['content'])
        is_valid, result = validate_arithmetic(thinking, nums)
        rewards.append(1.0 if is_valid else 0.0)
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for correct output format"""
    pattern = r"^<thinking>.*?</thinking>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def answer_match_reward_func(completions, answer, **kwargs) -> list[float]:
    """Additional reward if answer matches target (but not required)"""
    rewards = []
    for completion in completions:
        model_answer = extract_xml_answer(completion[0]['content'])
        try:
            matches = float(model_answer) == float(answer)
            rewards.append(1.0 if matches else 0.0)
        except:
            rewards.append(0.0)
    return rewards

# Training configuration
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=2e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=128,
    max_steps=2500,
    save_steps=500,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=str(run_dir),
)

# Initialize dataset and trainer
dataset = get_countdown_questions()
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,      # 0.5 for correct format
        arithmetic_reward_func,  # 1.0 for valid arithmetic
        answer_match_reward_func # 1.0 bonus for matching target
    ],
    args=training_args,
    train_dataset=dataset,
)

def test_model(prompt="Using these numbers [4, 5, 6], find a valid arithmetic result", nums=None):
    """Test model generation"""
    model_for_inference = FastLanguageModel.for_inference(model)

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)

    output = model_for_inference.generate(
        tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Validate generated response
    thinking = extract_xml_thinking(response)
    answer = extract_xml_answer(response)
    
    if nums:
        is_valid, result = validate_arithmetic(thinking, nums)
        print(f"Arithmetic valid: {is_valid}")
        print(f"Calculated result: {result}")
        
    return response

if __name__ == "__main__":
    try:
        print("Starting training...")
        print(f"Logs will be saved to: {run_dir}")
        trainer.train()

        print("Saving LoRA weights...")
        model.save_lora(str(run_dir / "arithmetic_lora"))

        # Test cases
        test_nums = [4, 5, 6]
        print("\nTesting model:")
        response = test_model(
            f"Find a valid arithmetic result using these numbers: {test_nums}",
            nums=test_nums
        )
        print(f"Response:\n{response}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
