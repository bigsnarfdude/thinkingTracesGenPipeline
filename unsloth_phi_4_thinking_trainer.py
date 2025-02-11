# gsm8k phi4 trainer latest working

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
        # Extract thinking part
        thinking = ""
        if "<thinking>" in response and "</thinking>" in response:
            thinking = response.split("<thinking>")[1].split("</thinking>")[0].strip()
        
        # Extract model's answer
        model_answer = ""
        if "<answer>" in response and "</answer>" in response:
            model_answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        
        # Create log entry
        log_entry = {
            "step": step,
            "question": question,
            "true_answer": true_answer,
            "thinking": thinking,
            "model_answer": model_answer,
            "full_response": response
        }
        
        # Save to JSON file
        log_file = run_dir / "interactions.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"Error logging interaction: {str(e)}")

# Low memory configuration
max_seq_length = 256
lora_rank = 8
gpu_memory_utilization = 0.7  # Increased to 0.7 as specified

print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model with memory optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",  # Changed to Phi-4
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gpu_memory_utilization,
)

# Minimal PEFT configuration for Phi-4
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["gate_proj", "up_proj", "down_proj"],  # Updated for Phi-4
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# System prompt using thinking tags
SYSTEM_PROMPT = """
You must respond in exactly this format, with no extra text before or after:

<thinking>
Show your step-by-step solution here.
Each step should be clear and complete.
Show all calculations.
</thinking>
<answer>
Your final numerical answer here, with no units or explanation.
</answer>

Example:
Question: If a train travels 60 mph for 2 hours, how far does it go?

<thinking>
1. Distance = Speed × Time
2. Speed = 60 mph
3. Time = 2 hours
4. Distance = 60 × 2 = 120 miles
</thinking>
<answer>
120
</answer>
"""

# Save configuration
with open(run_dir / "config.json", "w") as f:
    config = {
        "model": "unsloth/Phi-4",
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "gpu_memory_utilization": gpu_memory_utilization,
        "system_prompt": SYSTEM_PROMPT
    }
    json.dump(config, f, indent=2)

def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return ""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# Global step counter
training_step = 0

# Reward functions with logging
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    global training_step
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Log each interaction
    for response in responses:
        log_interaction(q, answer[0], response, training_step)
    training_step += 1
    
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<thinking>.*?</thinking>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<thinking>.*?</thinking>.*?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if "<thinking>" in text and "</thinking>" in text:
        count += 0.25
    if "<answer>" in text and "</answer>" in text:
        count += 0.25
    # Penalize extra text after closing tags
    final_tag_pos = max(text.rfind("</thinking>"), text.rfind("</answer>"))
    if final_tag_pos > 0:
        remaining_text = text[final_tag_pos:].strip()
        if remaining_text:
            count -= 0.001 * len(remaining_text)
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Low memory training configuration
training_args = GRPOConfig(
    use_vllm=False,  # Disabled vLLM
    learning_rate=5e-6,
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
    gradient_accumulation_steps=4,  # Increased for stability
    num_generations=4,  # Reduced from 8
    max_prompt_length=256,
    max_completion_length=128,  # Reduced from 200
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=str(run_dir),
)

# Initialize dataset and trainer
dataset = get_gsm8k_questions()
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

def test_model(prompt="How many r's are in strawberry?", with_lora=False):
    """Test model generation with and without LoRA"""
    # Prepare model for inference
    model_for_inference = FastLanguageModel.for_inference(model)
    
    # Apply chat template
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT} if with_lora else None,
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    
    # Generate response
    output = model_for_inference.generate(
        tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Log test outputs
    log_interaction(prompt, "N/A", response, "test")
    
    return response

if __name__ == "__main__":
    try:
        # Train model
        print("Starting training...")
        print(f"Logs will be saved to: {run_dir}")
        trainer.train()
        
        # Save LoRA weights
        print("Saving LoRA weights...")
        model.save_lora(str(run_dir / "grpo_saved_lora"))
        
        # Test model
        print("\nTesting without LoRA:")
        print(test_model(with_lora=False))
        
        print("\nTesting with LoRA:")
        print(test_model(with_lora=True))
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
