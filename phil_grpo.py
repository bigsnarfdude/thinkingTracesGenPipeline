from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType
import torch

# Dataset and Tokenizer Setup
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
dataset = dataset.shuffle(seed=42).select(range(50000))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

def generate_r1_prompt(numbers, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    },
    {
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
    },
    {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
    }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

# Process dataset
dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

def format_reward_func(completions, target, **kwargs):
    rewards = []
    for i, (completion, gt) in enumerate(zip(completions, target)):
        try:
            if "... </think>" in completion and not completion.startswith("<think>"):
                completion = "<think>" + completion
            
            if completion.strip().startswith("..."):
                completion = "<think>" + completion
            
            regex = r"^(?:<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>)$"
            think_count = completion.count("<think>")
            answer_count = completion.count("<answer>")
            
            match = re.search(regex, completion)
            reward = 1.0 if (match is not None and think_count == 1 and answer_count == 1) else 0.0
            rewards.append(reward)
        except Exception as e:
            rewards.append(0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            completion = "<think>" + completion
            match = re.search(r"<answer>(.*?)</answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            equation = match.group(1).strip()
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue
            
            result = eval(equation, {"__builtins__": None}, {})
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# Test samples for validation (testing code remains the same...)

# PEFT Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Training Configuration
training_args = GRPOConfig(
    output_dir="qwen-r1-aha-moment",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=1000,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    max_prompt_length=128,
    max_completion_length=256,
    num_generations=2,
    beta=0.1,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    warmup_steps=100,
)

# Initialize model first
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,
)

# Initialize trainer with model
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

# Train and save the model
trainer.train()
trainer.save_model(training_args.output_dir)
