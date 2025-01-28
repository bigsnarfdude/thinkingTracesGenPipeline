from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
from tqdm import tqdm
import json
import re
from datetime import datetime
import os
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_model(model_type):
    """Set up the model and tokenizer based on model type."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    try:
        if model_type == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(
                base_model, 
                "qwen-sft-lora-epoch1-final",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            model.eval()
            return model, tokenizer
            
        elif model_type == "base":
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            return model, tokenizer
            
        elif model_type == "sft":
            model_name = "YWZBrandon/openai-gsm8k_Qwen-Qwen2.5-1.5B_full_sft_2e-6"
            pipeline_obj = pipeline(
                "text-generation", 
                model=model_name, 
                device=device,
                trust_remote_code=True
            )
            return pipeline_obj, None
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error setting up model {model_type}: {str(e)}")
        raise

def generate_response(model, tokenizer, question, model_type, max_length=2048):
    """Generate model response for a given question."""
    try:
        if model_type == "sft":
            output = model(
                question,
                max_new_tokens=256,
                return_full_text=False
            )
            return output[0]['generated_text'] if isinstance(output, list) and output else ""

        system_msg = """You are a mathematical problem solver. Follow these rules EXACTLY:

1. Start your response with ONLY the final answer in \\boxed{} notation
2. Do NOT put $ signs around the \\boxed{} answer
3. After the answer, add a line break, then show your solution steps

IMPORTANT FORMAT RULES:
- Write fractions as \\frac{numerator}{denominator}
- Write coordinates as \\boxed{(x,y)} with parentheses
- Write pi as \\pi
- Keep ALL content inside ONE \\boxed{} at the start

Examples of CORRECT format:
\\boxed{42}
\\boxed{\\frac{1}{2}}
\\boxed{(\\sqrt{5}, \\frac{\\pi}{2})}

Examples of INCORRECT format:
$\\boxed{42}$  <- No dollar signs
\\boxed{1}, \\boxed{2}  <- Only one boxed answer
\\boxed{pi}  <- Use \\pi instead"""

        prompt = f"{system_msg}\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip() if response.startswith(prompt) else response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return ""

def extract_answer(text):
    """Extract answer from model response."""
    if not text:
        return ""
    
    def find_matching_brace(s, start):
        """Find the matching closing brace starting from position start."""
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i if count == 0 else -1

    # Clean up text and handle dollar signs
    text = text.strip()
    if text.startswith('$') and text.endswith('$'):
        text = text[1:-1].strip()

    # Look for \boxed{} pattern
    boxed_start = text.find('\\boxed{')
    if boxed_start != -1:
        content_start = boxed_start + 7
        end_pos = find_matching_brace(text, content_start)
        if end_pos != -1:
            content = text[content_start:end_pos-1]
            return f'\\boxed{{{content}}}'

    # Fallback patterns
    patterns = [
        (r'\$\\boxed{([^}]+)}\$', r'\boxed{\1}'),
        (r'\\boxed{([^}]+)}', r'\boxed{\1}'),
        (r'\\[\(\[]([^\[\]\(\)]+?)\\[\)\]]', r'\boxed{\1}')
    ]

    for pattern, template in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return template.replace('\\1', content)

    return ""

def normalize_answer(answer):
    """Normalize answer string for comparison."""
    if not answer:
        return ""
    
    # Remove whitespace and convert to lowercase
    answer = re.sub(r'\s+', '', answer.lower())
    
    # Remove LaTeX enclosures (\left, \right, \boxed, etc.)
    answer = re.sub(r'\\left', '', answer)
    answer = re.sub(r'\\right', '', answer)
    answer = re.sub(r'\\boxed{(.+?)}', r'\1', answer)
    answer = re.sub(r'\\[\(\[](.+?)\\[\)\]]', r'\1', answer)
    
    # Handle coordinates - standardize parentheses and comma spacing
    answer = re.sub(r'\(([^)]+)\)', lambda m: '(' + re.sub(r'\s*,\s*', ',', m.group(1)) + ')', answer)
    
    # Normalize fractions
    def normalize_fraction(match):
        num = match.group(1)
        den = match.group(2)
        # Convert to decimal for numerical fractions
        if num.isdigit() and den.isdigit():
            return str(int(num)/int(den))
        return f"{num}/{den}"
    answer = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', normalize_fraction, answer)
    
    # Normalize special values
    answer = answer.replace('\\pi', 'pi')
    answer = answer.replace('\\infty', 'infinity')
    answer = re.sub(r'\\(?:circ|degree)s?', 'degrees', answer)
    
    # Handle variables and operators
    answer = re.sub(r'(\d)([a-z])', r'\1*\2', answer)  # Add * between numbers and variables
    answer = answer.replace('^', '**')  # Standardize exponentiation
    
    # Remove remaining LaTeX commands and unnecessary brackets
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[{}\[\]](?![0-9])', '', answer)
    
    # Keep parentheses for coordinates and grouping
    # but remove standalone parentheses
    if not ('(' in answer and ')' in answer and ',' in answer):
        answer = re.sub(r'[()]', '', answer)
    
    return answer

def answers_match(predicted, reference):
    """Compare predicted answer with reference answer with detailed logging."""
    if not predicted or not reference:
        return False
    
    norm_predicted = normalize_answer(predicted)
    norm_reference = normalize_answer(reference)
    
    # Log normalization results for debugging
    logger.debug(f"Original predicted: {predicted}")
    logger.debug(f"Original reference: {reference}")
    logger.debug(f"Normalized predicted: {norm_predicted}")
    logger.debug(f"Normalized reference: {norm_reference}")
    
    # Try exact match first
    if norm_predicted == norm_reference:
        return True
    
    # For coordinate pairs, check numerical equivalence
    coord_pattern = r'\(([\d\.\-]+),([\d\.\-]+)\)'
    pred_coords = re.match(coord_pattern, norm_predicted)
    ref_coords = re.match(coord_pattern, norm_reference)
    
    if pred_coords and ref_coords:
        try:
            pred_x, pred_y = map(float, pred_coords.groups())
            ref_x, ref_y = map(float, ref_coords.groups())
            return abs(pred_x - ref_x) < 1e-6 and abs(pred_y - ref_y) < 1e-6
        except ValueError:
            pass
    
    return False

def evaluate_model(model_type, num_samples=50):
    """Evaluate model on MATH-500 dataset."""
    try:
        # Load dataset
        dataset = load_dataset("HuggingFaceH4/MATH-500")['test']
        if num_samples:
            dataset = dataset.select(range(num_samples))
        
        # Setup model
        model, tokenizer = setup_model(model_type)
        
        results = []
        correct = 0
        total = 0
        
        logger.info(f"Evaluating {model_type} model on {num_samples if num_samples else 'all'} samples...")
        
        for idx, sample in enumerate(tqdm(dataset)):
            try:
                problem = sample['problem']
                true_answer = sample['answer']
                
                formatted_question = (
                    "Solve this math problem. Express your answer using LaTeX notation with \\boxed{}.\n\n"
                    f"Problem: {problem}"
                )
                
                model_response = generate_response(model, tokenizer, formatted_question, model_type)
                predicted_answer = extract_answer(model_response)
                
                # Log first few samples
                if idx < 3:
                    logger.info(f"\nSample {idx}:")
                    logger.info(f"Problem: {problem[:200]}...")
                    logger.info(f"True answer: {true_answer}")
                    logger.info(f"Model response: {model_response[:200]}...")
                    logger.info(f"Extracted answer: {predicted_answer}")
                
                is_correct = answers_match(predicted_answer, true_answer)
                correct += int(is_correct)
                total += 1
                
                results.append({
                    'problem': problem,
                    'true_answer': true_answer,
                    'predicted_answer': predicted_answer,
                    'full_response': model_response,
                    'is_correct': is_correct,
                    'subject': sample['subject'],
                    'level': sample['level']
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"\nProcessed {total} samples successfully")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        
        return {
            'model_type': model_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'detailed_results': results
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise


def save_results(results, all_results=None):
    """Save evaluation results to files with combined summary."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"math500_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize or update all_results
        if all_results is None:
            all_results = []
        
        # Create current model summary
        current_summary = {
            'model_type': results['model_type'],
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'timestamp': timestamp
        }
        
        # Add to all results
        all_results.append(current_summary)
        
        # Save combined summary of all models
        with open(f"{output_dir}/summary_all_models.json", 'w') as f:
            json.dump({
                'models': all_results,
                'timestamp': timestamp
            }, f, indent=2)
        
        # Save individual model summary
        with open(f"{output_dir}/summary_{results['model_type']}.json", 'w') as f:
            json.dump(current_summary, f, indent=2)
        
        # Save detailed results
        with open(f"{output_dir}/detailed_{results['model_type']}.jsonl", 'w', encoding='utf-8') as f:
            for result in results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        logger.info(f"Results saved to {output_dir}")
        return output_dir, all_results
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    """Main entry point."""
    try:
        logger.info("Starting evaluation...")
        NUM_SAMPLES = 50
        model_types = ["base", "lora", "sft"]
        all_results = []
        
        for model_type in model_types:
            try:
                logger.info(f"\nStarting setup for {model_type} model...")
                
                # Setup model
                logger.info(f"Setting up {model_type} model...")
                model, tokenizer = setup_model(model_type)
                logger.info(f"Successfully set up {model_type} model")
                
                # Evaluate model
                logger.info(f"Beginning evaluation of {model_type} model...")
                results = evaluate_model(model_type, NUM_SAMPLES)
                logger.info(f"Completed evaluation of {model_type} model")
                
                # Save results
                logger.info(f"Saving results for {model_type} model...")
                output_dir, all_results = save_results(results, all_results)
                
                logger.info(f"\nResults for {model_type} model:")
                logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")
                logger.info(f"Correct: {results['correct']}/{results['total']}")
                logger.info(f"Results saved in: {output_dir}")
                
            except Exception as e:
                logger.error(f"Error processing {model_type} model: {str(e)}")
                logger.error(f"Traceback:", exc_info=True)
                continue  # Continue with next model even if one fails
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler('evaluation.log')  # File handler
        ]
    )
    
    # Suppress TensorFlow and other warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    
    main()
