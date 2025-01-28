from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from tqdm import tqdm
import json
import re
from datetime import datetime
import os
from datasets import load_dataset

class ModelEvaluator:
    def __init__(self, model_type):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._setup_model()

    def _setup_model(self):
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        if self.model_type == "lora":
            # LoRA model setup
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, "qwen-sft-lora-epoch1-final")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
        elif self.model_type == "base":
            # Base model setup
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
        elif self.model_type == "sft":
            # SFT model setup
            model_name = "YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6"
            pipeline_obj = pipeline("text-generation", model=model_name, device=self.device)
            return pipeline_obj, None
            
        return model, tokenizer

    def generate_response(self, question, max_length=2048):
        if self.model_type == "sft":
            output = self.model(
                [{"role": "user", "content": question}],
                max_new_tokens=256,
                return_full_text=False
            )[0]
            return output['generated_text']
            
        # For base and LoRA models
        system_msg = """You are a mathematical problem solver. Structure your response in two parts:
        1. Show your thinking process between <thinking> and </thinking> tags
        2. Present your solution between <solution> and </solution> tags
        3. Present your final answer between <answer> and </answer> tags

        The answer should be in LaTeX format and match the format shown in the question.
        """
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

def extract_sections(text):
    """Extract thinking, solution sections and final answer from the response."""
    thinking = ""
    solution = ""
    answer = ""
    
    # Extract thinking section
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    # Extract solution section
    solution_match = re.search(r'<solution>(.*?)</solution>', text, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
    
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return thinking, solution, answer

def normalize_answer(answer):
    """Normalize answer string for comparison."""
    # Remove whitespace and convert to lowercase
    answer = re.sub(r'\s+', '', answer.lower())
    # Remove unnecessary characters but preserve important math symbols
    answer = re.sub(r'[,\{\}\[\]()](?![0-9])', '', answer)
    # Normalize LaTeX fractions
    answer = re.sub(r'\\frac{(\d+)}{(\d+)}', r'\1/\2', answer)
    # Normalize pi
    answer = answer.replace('\\pi', 'pi')
    return answer

def is_correct_answer(predicted, reference):
    """Compare predicted answer with reference answer."""
    if not predicted or not reference:
        return False
    return normalize_answer(predicted) == normalize_answer(reference)

def evaluate_model(model_type, num_samples=50):
    """Evaluate a specific model type on MATH-500 dataset."""
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500")['test']
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    # Initialize model
    evaluator = ModelEvaluator(model_type)
    
    results = []
    correct = 0
    total = 0
    
    print(f"\nEvaluating {model_type} model on {num_samples if num_samples else 'all'} samples...")
    
    for idx, sample in enumerate(tqdm(dataset)):
        try:
            # Get problem and answer
            problem = sample['problem']
            true_answer = sample['answer']
            
            # Format the question with LaTeX math mode hints
            formatted_question = (
                "Solve this math problem. Express your answer in the same format as shown in the question.\n\n"
                f"Problem: {problem}\n\n"
                "Show your work step by step, and use LaTeX notation for mathematical expressions."
            )
            
            # Generate response
            model_response = evaluator.generate_response(formatted_question)
            thinking, solution, predicted_answer = extract_sections(model_response)
            
            # Debug first sample
            if idx == 0:
                print("\nFirst problem processing:")
                print(f"Problem: {problem[:200]}")
                print(f"True answer: {true_answer}")
                print(f"Predicted answer: {predicted_answer}")
                print(f"Thinking: {thinking[:200] if thinking else 'None'}")
            
            # Check if correct
            is_correct = is_correct_answer(predicted_answer, true_answer)
            correct += int(is_correct)
            total += 1
            
            # Store result
            results.append({
                'problem': problem,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'thinking': thinking,
                'solution': solution,
                'full_response': model_response,
                'is_correct': is_correct,
                'subject': sample['subject'],
                'level': sample['level']
            })
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nProcessed {total} samples successfully")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'detailed_results': results
    }

def save_results(results):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"math500_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary metrics
    summary = {
        'model_type': results['model_type'],
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'timestamp': timestamp
    }
    
    with open(f"{output_dir}/summary_{results['model_type']}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(f"{output_dir}/detailed_{results['model_type']}.jsonl", 'w', encoding='utf-8') as f:
        for result in results['detailed_results']:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Additional analysis by subject and level
    subject_performance = {}
    level_performance = {}
    
    for result in results['detailed_results']:
        # Subject analysis
        subject = result['subject']
        if subject not in subject_performance:
            subject_performance[subject] = {'correct': 0, 'total': 0}
        subject_performance[subject]['total'] += 1
        if result['is_correct']:
            subject_performance[subject]['correct'] += 1
            
        # Level analysis
        level = result['level']
        if level not in level_performance:
            level_performance[level] = {'correct': 0, 'total': 0}
        level_performance[level]['total'] += 1
        if result['is_correct']:
            level_performance[level]['correct'] += 1
    
    # Calculate percentages and save analysis
    analysis = {
        'subject_performance': {
            subject: {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
            for subject, stats in subject_performance.items()
        },
        'level_performance': {
            level: {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
            for level, stats in level_performance.items()
        }
    }
    
    with open(f"{output_dir}/analysis_{results['model_type']}.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return output_dir

def main():
    # Number of samples to evaluate (set to None for full dataset)
    NUM_SAMPLES = 50
    
    # Models to evaluate - removed sky from the list
    model_types = ["base", "lora", "sft"]
    
    for model_type in model_types:
        print(f"\nEvaluating {model_type} model...")
        results = evaluate_model(model_type, NUM_SAMPLES)
        output_dir = save_results(results)
        
        print(f"\nResults for {model_type} model:")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Correct: {results['correct']}/{results['total']}")
        print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
