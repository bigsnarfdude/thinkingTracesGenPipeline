from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import json
import re

def setup_base_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=4096):
    # System message with correct tag format
    system_msg = """Structure your response in two parts:
    1. Show your thinking process between <thinking> and </thinking> tags
    2. Present your solution between <solution> and </solution> tags
    Include detailed reasoning and use LaTeX for mathematical expressions."""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

def evaluate_base_model():
    model, tokenizer = setup_base_model()
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    
    # Sample size for evaluation
    sample_size = 50
    samples = dataset.select(range(sample_size))
    
    results = []
    print(f"Evaluating base model on {sample_size} samples...")
    
    for sample in tqdm(samples):
        try:
            # Extract question and reference response
            question = next(conv['value'] for conv in sample['conversations'] if conv['from'] == 'user')
            reference = next(conv['value'] for conv in sample['conversations'] if conv['from'] == 'assistant')
            
            model_response = generate_response(model, tokenizer, question)
            
            # Extract thinking and solution sections using correct tags
            def extract_sections(text):
                thinking = ""
                solution = ""
                
                # Extract thinking section
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
                if thinking_match:
                    thinking = thinking_match.group(1).strip()
                
                # Extract solution section
                solution_match = re.search(r'<solution>(.*?)</solution>', text, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                
                return thinking, solution
            
            model_thinking, model_solution = extract_sections(model_response)
            ref_thinking, ref_solution = extract_sections(reference)
            
            # Extract boxed answer if present
            def extract_boxed_answer(text):
                match = re.search(r'\\boxed{([^}]+)}', text)
                return match.group(1) if match else None
            
            result = {
                "question": question,
                "model_response": {
                    "thinking": model_thinking,
                    "solution": model_solution,
                    "boxed_answer": extract_boxed_answer(model_response),
                    "full_response": model_response
                },
                "reference": {
                    "thinking": ref_thinking,
                    "solution": ref_solution,
                    "boxed_answer": extract_boxed_answer(reference),
                    "full_response": reference
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    output_file = "base_model_evaluation.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nEvaluation complete. Results saved to {output_file}")
    print(f"Processed {len(results)} samples successfully")

if __name__ == "__main__":
    evaluate_base_model()
