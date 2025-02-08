import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)

import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Low memory configuration
max_seq_length = 256
lora_rank = 8
gpu_memory_utilization = 0.6

print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model with memory optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gpu_memory_utilization,
)

# Minimal PEFT configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# System prompt
SYSTEM_PROMPT = """
Respond in the following format:
<thinking>
...
</thinking>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

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

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
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
    output_dir="outputs",
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
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT} if with_lora else None,
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    
    # Since vLLM is disabled, use standard generation
    output = model.generate(
        tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    try:
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save LoRA weights
        print("Saving LoRA weights...")
        model.save_lora("grpo_saved_lora")
        
        # Test model
        print("\nTesting without LoRA:")
        print(test_model(with_lora=False))
        
        print("\nTesting with LoRA:")
        print(test_model(with_lora=True))
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
