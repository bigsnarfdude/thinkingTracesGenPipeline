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
    max_seq_length = 2048
    lora_rank = 8
    gpu_memory_utilization = 0.7

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    model.load_lora("training_logs/run_20250209_061540/grpo_saved_lora")
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def create_prompt(sample):
    content = f"""Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. Show your step-by-step solution inside <think> </think> tags, and then give only the final equation inside <answer> </answer> tags.

Example format:
<think>
1. First, let's try...
2. Next...
3. Therefore...
</think>
<answer>
(1 + 2) / 3
</answer>"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. First think step by step about the solution, then provide the final equation."},
        {"role": "user", "content": content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def clean_response(response: str) -> str:
    """Clean the response by removing instruction artifacts"""
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    think_start = response.find("<think>")
    answer_end = response.rfind("</answer>")
    
    if think_start >= 0 and answer_end >= 0:
        response = response[think_start:answer_end + 9]
        
    return response.strip()

def extract_content(response: str, tag: str) -> str:
    """Extract content between specified tags"""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    start = response.find(start_tag)
    end = response.find(end_tag)
    
    if start >= 0 and end >= 0:
        content = response[start + len(start_tag):end]
        return content.strip()
    return ""

def evaluate_response(response, sample_info):
    cleaned_response = clean_response(response)
    thinking_content = extract_content(cleaned_response, "think")
    answer_content = extract_content(cleaned_response, "answer")
    
    metrics = {
        'has_think_tags': '<think>' in cleaned_response and '</think>' in cleaned_response,
        'has_answer_tags': '<answer>' in cleaned_response and '</answer>' in cleaned_response,
        'format_correct': False,
        'answer_present': bool(answer_content),
        'reasoning_present': len(thinking_content.split()) > 20,
        'nums': sample_info['nums'],
        'target': sample_info['target'],
        'full_response': response,
        'thinking': thinking_content,
        'answer': answer_content
    }
    
    if metrics['has_think_tags'] and metrics['has_answer_tags']:
        think_pos = cleaned_response.find('<think>')
        answer_pos = cleaned_response.find('<answer>')
        metrics['format_correct'] = think_pos < answer_pos
    
    return metrics

def save_results(results, filename="phi4_countdown_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename="phi4_countdown_results.json"):
    if Path(filename).exists():
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def process_batch(model, tokenizer, samples):
    """Process a batch of samples"""
    batch_prompts = [create_prompt(sample) for sample in samples]
    
    inputs = tokenizer(batch_prompts, 
                      return_tensors="pt", 
                      padding=True, 
                      truncation=True).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        min_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) 
                for output in outputs]
    
    return [evaluate_response(response, sample) 
            for response, sample in zip(responses, samples)]

def main():
    global model, tokenizer
    print("Setting up model...")
    model, tokenizer = setup_model()
    
    print("Loading dataset...")
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4-Unique", split="train")
    
    # Debug first with one sample
    sample = dataset[0]
    test_metrics = process_batch(model, tokenizer, [sample])[0]
    
    print("\n=== Test Sample ===")
    print(f"Numbers: {test_metrics['nums']}")
    print(f"Target: {test_metrics['target']}")
    print(f"Answer: {test_metrics['answer']}")
    print("\nMetrics:", json.dumps({k: v for k, v in test_metrics.items() 
                                  if k not in ['full_response', 'thinking']}, indent=2))
    
    response = input("\nContinue with batch processing? (y/n): ")
    if response.lower() != 'y':
        return
    
    N = 1000
    BATCH_SIZE = 5
    results = load_results()
    
    if results:
        print(f"Loaded {len(results)} existing results")
        if len(results) >= N:
            print("Already have enough results")
            return results
    
    samples_needed = N - len(results)
    if samples_needed > 0:
        print(f"\nProcessing {samples_needed} more samples...")
        # Convert numpy indices to python int
        selected_indices = [int(i) for i in np.random.choice(len(dataset), 
                                                           size=min(samples_needed, len(dataset)), 
                                                           replace=False)]
        
        for batch_start in tqdm(range(0, len(selected_indices), BATCH_SIZE), 
                               desc="Processing batches"):
            batch_end = min(batch_start + BATCH_SIZE, len(selected_indices))
            batch_indices = selected_indices[batch_start:batch_end]
            # Convert indices to int before accessing dataset
            batch_samples = [dataset[int(i)] for i in batch_indices]
            
            batch_metrics = process_batch(model, tokenizer, batch_samples)
            
            # Print first response in batch
            if batch_metrics:
                print("\nSample from current batch:")
                print(f"Numbers: {batch_metrics[0]['nums']}")
                print(f"Target: {batch_metrics[0]['target']}")
                print(f"Answer: {batch_metrics[0]['answer']}")
            
            results.extend(batch_metrics)
            save_results(results)
            
            if len(results) >= N:
                break
    
    total = len(results)
    aggregates = {
        'model': 'Phi-4 with LoRA',
        'total_samples': total,
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
    
    with open("phi4_countdown_metrics.json", 'w') as f:
        json.dump(aggregates, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()
