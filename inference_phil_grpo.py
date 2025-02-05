from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="mps",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "qwen-r1-aha-moment")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def generate_response(numbers, target):
    prompt = [{
        "role": "system",
        "content": "You must complete both thinking and answer sections. Always end with <answer> tag containing the final equation."
    },
    {
        "role": "user", 
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags."
    },
    {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
    }]
    
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Example usage
numbers = [19, 36, 55, 7]
target = 65
response = generate_response(numbers, target)
print(response)
