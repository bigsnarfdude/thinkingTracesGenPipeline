from datasets import load_dataset
from transformers import pipeline
import torch
import re
from tqdm import tqdm
import json
import csv
from datetime import datetime
import os

def extract_answer(text):
    """Extract the final answer (number after ####) from the model output."""
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def save_results(results, model_name):
    """Save results to files in a timestamped directory."""
    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short_name = model_name.split('/')[-1]
    output_dir = f"gsm8k_results_{model_short_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save summary metrics as JSON
    summary = {
        'model_name': model_name,
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'timestamp': timestamp
    }
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results as CSV
    with open(f"{output_dir}/detailed_results.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'true_answer', 'predicted_answer', 'is_correct', 'full_response'])
        for result in results['detailed_results']:
            writer.writerow([
                result['question'],
                result['true_answer'],
                result['predicted_answer'],
                result['is_correct'],
                result['full_response']
            ])

    # Save incorrect predictions for analysis
    incorrect_results = [r for r in results['detailed_results'] if not r['is_correct']]
    with open(f"{output_dir}/incorrect_predictions.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'true_answer', 'predicted_answer', 'full_response'])
        for result in incorrect_results:
            writer.writerow([
                result['question'],
                result['true_answer'],
                result['predicted_answer'],
                result['full_response']
            ])

    return output_dir

def evaluate_gsm8k(model_name="YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6", 
                   split="test", 
                   num_samples=None):
    # Load dataset
    dataset = load_dataset("gsm8k", 'main')[split]
    if num_samples:
        dataset = dataset.select(range(num_samples))

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline("text-generation", 
                       model=model_name,
                       device=device)

    # Evaluation metrics
    correct = 0
    total = 0
    results = []

    # Process each example
    for example in tqdm(dataset):
        try:
            # Generate response
            question = example['question']
            true_answer = extract_answer(example['answer'])
            
            output = generator(
                [{"role": "user", "content": question}],
                max_new_tokens=256,
                return_full_text=False
            )[0]
            
            predicted_answer = extract_answer(output['generated_text'])
            
            # Check if correct
            is_correct = (predicted_answer is not None and predicted_answer == true_answer)
            correct += int(is_correct)
            total += 1

            # Store result
            results.append({
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'full_response': output['generated_text'],
                'is_correct': is_correct
            })

        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'detailed_results': results
    }

if __name__ == "__main__":
    # Set parameters
    MODEL_NAME = "YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6"
    NUM_SAMPLES = 10  # Set to None for full dataset

    # Run evaluation
    print(f"Evaluating model: {MODEL_NAME}")
    print(f"Number of samples: {NUM_SAMPLES if NUM_SAMPLES else 'Full dataset'}")
    
    results = evaluate_gsm8k(model_name=MODEL_NAME, num_samples=NUM_SAMPLES)
    
    # Print summary
    print(f"\nAccuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    
    # Save results
    output_dir = save_results(results, MODEL_NAME)
    print(f"\nResults saved in directory: {output_dir}")
