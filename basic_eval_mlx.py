import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from mlx_lm import load
from eval import evaluate_model, generate_batch, find_last_number

def evaluate_models(model_list, test_data_path="test.jsonl", output_dir="eval_results"):
    """
    Evaluate multiple models on the same test set and save detailed results.
    
    Args:
        model_list: List of model identifiers (paths or huggingface ids)
        test_data_path: Path to the test data in JSONL format
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    with open(test_data_path, "r") as f:
        rows = [json.loads(line) for line in f]
        questions = [row["question"] for row in rows]
        answers = [row["answer"].split("#### ")[1] for row in rows]
    
    # Prepare results container
    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "questions": questions,
        "answers": answers,
        "per_question_results": [{
            "id": i,
            "question": q,
            "answer": a,
            "model_predictions": {}
        } for i, (q, a) in enumerate(zip(questions, answers))]
    }
    
    # Evaluate each model
    for model_name in model_list:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        # Load model and tokenizer
        model, tokenizer = load(model_name)
        
        # Prepare detailed results tracking
        model_responses = []
        model_predictions = []
        model_correctness = []
        
        # Loop through data in batches
        batch_size = 16  # Smaller batch size for detailed tracking
        num_batches = (len(questions) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(questions))
            
            # Prepare batch
            batch_questions = questions[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            
            # Format prompts with chat template
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=False
                ) for q in batch_questions
            ]
            
            # Generate responses
            responses = generate_batch(model, tokenizer, prompts, verbose=False)
            
            # Extract predictions
            batch_predictions = [find_last_number(text) for text in responses]
            
            # Calculate correctness
            batch_correctness = [
                1 if pred is not None and pred == int(ans.replace(',', '')) else 0
                for pred, ans in zip(batch_predictions, batch_answers)
            ]
            
            # Store results
            model_responses.extend(responses)
            model_predictions.extend(batch_predictions)
            model_correctness.extend(batch_correctness)
        
        # Calculate final accuracy
        accuracy = sum(model_correctness) / len(model_correctness)
        print(f"Model {model_name} accuracy: {accuracy:.2%}")
        
        # Store model results
        all_results["models"][model_name] = {
            "accuracy": accuracy,
            "responses": model_responses,
            "predictions": [str(p) if p is not None else "None" for p in model_predictions],
            "correctness": model_correctness
        }
        
        # Update per-question results
        for i, (resp, pred, correct) in enumerate(zip(model_responses, model_predictions, model_correctness)):
            all_results["per_question_results"][i]["model_predictions"][model_name] = {
                "response": resp,
                "prediction": str(pred) if pred is not None else "None",
                "correct": correct
            }
    
    # Save full results as JSON
    results_file = os.path.join(output_dir, f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary CSV for easy comparison
    summary_data = []
    for i, q_result in enumerate(all_results["per_question_results"]):
        row = {
            "question_id": i,
            "question": q_result["question"],
            "correct_answer": q_result["answer"]
        }
        
        # Add each model's prediction and correctness
        for model_name in model_list:
            if model_name in q_result["model_predictions"]:
                row[f"{model_name}_prediction"] = q_result["model_predictions"][model_name]["prediction"]
                row[f"{model_name}_correct"] = q_result["model_predictions"][model_name]["correct"]
        
        summary_data.append(row)
    
    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Generate model comparison table
    model_comparison = {
        "model": [],
        "accuracy": [],
        "correct_count": [],
        "total_questions": len(questions)
    }
    
    for model_name in model_list:
        model_comparison["model"].append(model_name)
        model_comparison["accuracy"].append(all_results["models"][model_name]["accuracy"])
        model_comparison["correct_count"].append(sum(all_results["models"][model_name]["correctness"]))
    
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df = comparison_df.sort_values("accuracy", ascending=False)
    comparison_file = os.path.join(output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    print(f"\nFull results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Model comparison saved to: {comparison_file}")
    
    return all_results

if __name__ == "__main__":
    # List of models to evaluate
    models = [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        # Add more models here
        # "path/to/model2",
        # "huggingface-org/model3",
    ]
    
    # Run evaluation
    evaluate_models(models)
