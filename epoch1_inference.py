from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from tqdm import tqdm
import json
import re

def setup_model():
    print("Loading base model and tokenizer...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading LoRA weights...")
    # Load the trained LoRA weights
    model = PeftModel.from_pretrained(base_model, "qwen-lora-epoch1-final")
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=2048):
    # System message requesting structured output
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
    
    with torch.no_grad():
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

def extract_sections(text):
    """Extract thinking and solution sections from the response."""
    thinking = ""
    solution = ""
    
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    solution_match = re.search(r'<solution>(.*?)</solution>', text, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
    
    return thinking, solution

def run_inference(num_samples=50):
    # Load model and tokenizer
    model, tokenizer = setup_model()
    
    # Load test dataset
    print("Loading test dataset...")
    from datasets import load_dataset
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    samples = dataset.select(range(num_samples))
    
    results = []
    print(f"Running inference on {num_samples} samples...")
    
    for sample in tqdm(samples):
        try:
            # Extract question from sample
            question = next(conv['value'] for conv in sample['conversations'] if conv['from'] == 'user')
            reference = next(conv['value'] for conv in sample['conversations'] if conv['from'] == 'assistant')
            
            # Generate response
            model_response = generate_response(model, tokenizer, question)
            
            # Extract sections
            model_thinking, model_solution = extract_sections(model_response)
            ref_thinking, ref_solution = extract_sections(reference)
            
            # Store results
            result = {
                "question": question,
                "model_output": {
                    "thinking": model_thinking,
                    "solution": model_solution,
                    "full_response": model_response
                },
                "reference": {
                    "thinking": ref_thinking,
                    "solution": ref_solution,
                    "full_response": reference
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # Save results
    output_file = "epoch1_inference_results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nInference complete. Results saved to {output_file}")
    print(f"Processed {len(results)} samples successfully")

if __name__ == "__main__":
    run_inference()
