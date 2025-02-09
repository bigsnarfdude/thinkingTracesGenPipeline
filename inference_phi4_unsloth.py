import torch
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 256
lora_rank = 8
gpu_memory_utilization = 0.7

# System prompt
SYSTEM_PROMPT = """
You must respond in exactly this format, with no extra text before or after:

<thinking>
Show your step-by-step solution here.
Each step should be clear and complete.
Show all calculations.
</thinking>
<answer>
Your final numerical answer here, with no units or explanation.
</answer>
"""

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gpu_memory_utilization,
)

# Load LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load the trained LoRA weights
model.load_lora("training_logs/run_20250209_061540/grpo_saved_lora")

# Prepare model for inference
model = FastLanguageModel.for_inference(model)

def generate_answer(prompt: str) -> str:
    """Generate an answer for a given math problem"""
    # Apply chat template
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    
    # Generate response
    output = model.generate(
        tokenizer(text, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    test_questions = [
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "John has 5 apples and buys 3 more. How many apples does he have?",
    ]
    
    for question in test_questions:
        print("\nQuestion:", question)
        print("\nResponse:", generate_answer(question))
