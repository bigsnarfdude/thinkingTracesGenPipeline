from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "qwen-r1-aha-moment"  # Your saved model path
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def generate_response(numbers, target):
    prompt = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    },
    {
        "role": "user", 
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags."
    }]
    
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=384,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
numbers = [19, 36, 55, 7]
target = 65
response = generate_response(numbers, target)
print(response)
