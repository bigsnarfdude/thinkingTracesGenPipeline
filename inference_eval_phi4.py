import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import re
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

def setup_model():
    # Configuration
    max_seq_length = 256
    lora_rank = 8
    gpu_memory_utilization = 0.7

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Load LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load the trained LoRA weights
    model.load_lora("training_logs/run_20250209_061540/grpo_saved_lora")

    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def create_prompt(sample):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def evaluate_response(response, sample_info):
    metrics = {
        'has_think_tags': False,
        'has_answer_tags': False,
        'format_correct': False,
        'answer_present': False,
        'reasoning_present': False,
        'nums': sample_info['nums'],
        'target': sample_info['target'],
        'full_response': response
    }
    
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    metrics['has_think_tags'] = bool(think_match)
    if think_match:
        think_content = think_match.group(1).strip()
        metrics['reasoning_present'] = len(think_content) > 50
        metrics['thinking'] = think_content
    
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    metrics['has_answer_tags'] = bool(answer_match)
    
    if metrics['has_think_tags'] and metrics['has_answer_tags']:
        think_pos = response.find('<think>')
        answer_pos = response.find('<answer>')
        metrics['format_correct'] = think_pos < answer_pos
    
    if answer_match:
        answer = answer_match.group(1).strip()
        metrics['answer_present'] = bool(answer)
        metrics['answer'] = answer
    
    return metrics

def save_results(results, filename="phi4_countdown_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename="phi4_countdown_results.json"):
    if Path(filename).exists():
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def main():
    global model, tokenizer  # Make them global for create_prompt to access
    model, tokenizer = setup_model()
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4-Unique", split="train")
    
    N = 1000
    BATCH_SIZE = 10
    
    # Load existing results or start fresh
    results = load_results()
    if results:
        print(f"Loaded {len(results)} existing results")
        if len(results) >= N:
            print("Already have enough results")
            return results
    
    # Get indices we haven't processed yet
    processed_nums = {tuple(r['nums']) for r in results}
    remaining_samples = [(i, sample) for i, sample in enumerate(dataset) 
                        if tuple(sample['nums']) not in processed_nums]
    
    # Randomly sample from remaining
    samples_needed = N - len(results)
    if samples_needed > 0:
        selected_samples = np.random.choice(len(remaining_samples), 
                                          size=min(samples_needed, len(remaining_samples)), 
                                          replace=False)
        
        for batch_start in tqdm(range(0, len(selected_samples), BATCH_SIZE), 
                               desc="Processing batches"):
            batch_indices = selected_samples[batch_start:batch_start + BATCH_SIZE]
            batch_samples = [remaining_samples[i][1] for i in batch_indices]
            batch_prompts = [create_prompt(sample) for sample in batch_samples]
            
            # Generate responses
            inputs = tokenizer(batch_prompts, 
                             return_tensors="pt", 
                             padding=True, 
                             truncation=True).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.95
            )
            
            responses = [tokenizer.decode(output, skip_special_tokens=True) 
                        for output in outputs]
            
            # Evaluate and save batch results
            batch_metrics = [evaluate_response(response, sample) 
                           for response, sample in zip(responses, batch_samples)]
            results.extend(batch_metrics)
            
            # Save after each batch
            save_results(results)
            
            # Print progress
            print(f"\nProcessed {len(results)}/{N} samples")
            if len(results) >= N:
                break
    
    # Calculate and save final metrics
    total = len(results)
    aggregates = {
        'model': 'Phi-4 with LoRA',
        'think_tags_present': sum(r['has_think_tags'] for r in results) / total * 100,
        'answer_tags_present': sum(r['has_answer_tags'] for r in results) / total * 100,
        'correct_format': sum(r['format_correct'] for r in results) / total * 100,
        'answer_present': sum(r['answer_present'] for r in results) / total * 100,
        'reasoning_present': sum(r['reasoning_present'] for r in results) / total * 100
    }
    
    print("\nFinal Aggregate Metrics:")
    for metric, value in aggregates.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.1f}%")
        else:
            print(f"{metric}: {value}")
    
    # Save aggregates
    with open("phi4_countdown_metrics.json", 'w') as f:
        json.dump(aggregates, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()
