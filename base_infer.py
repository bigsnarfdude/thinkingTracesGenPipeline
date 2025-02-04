from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from random import randint

def load_model():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

def generate_solution(model, tokenizer, nums, target):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    sample = dataset[randint(0, len(dataset))]
    
    response = generate_solution(model, tokenizer, sample['nums'], sample['target'])
    print(response)
