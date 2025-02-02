from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType
import torch
from typing import List, Dict, Any
from equation_generator import EquationPromptGenerator, PromptConfig

class RewardFunctions:
    """Class containing reward functions for evaluating model outputs"""
    
    @staticmethod
    def format_reward_func(completions: List[str], target: List[Any], **kwargs) -> List[float]:
        """Check if the completion follows the correct format"""
        rewards = []
        for completion in completions:
            try:
                if "... </think>" in completion and not completion.startswith("<think>"):
                    completion = "<think>" + completion
                
                if completion.strip().startswith("..."):
                    completion = "<think>" + completion
                
                regex = r"^(?:<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>)$"
                think_count = completion.count("<think>")
                answer_count = completion.count("<answer>")
                
                reward = 1.0 if (re.search(regex, completion) and think_count == 1 and answer_count == 1) else 0.0
                rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def equation_reward_func(completions: List[str], target: List[Any], nums: List[List[int]], **kwargs) -> List[float]:
        """Check if the equation is valid and gives the correct result"""
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
                rewards.append(1.0 if abs(float(result) - float(gt)) < 1e-5 else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

class ModelTrainer:
    """Class for managing the model training process"""
    
    def __init__(self, model_name: str, dataset_id: str, num_samples: int = 50000):
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.num_samples = num_samples
        
        # Initialize components
        self.setup_dataset()
        self.setup_prompt_generator()
        self.setup_model()
        self.setup_training_config()
        
    def setup_dataset(self):
        """Setup and preprocess the dataset"""
        dataset = load_dataset(self.dataset_id, split="train")
        dataset = dataset.shuffle(seed=42).select(range(self.num_samples))
        self.dataset = dataset
        
    def setup_prompt_generator(self):
        """Initialize the prompt generator"""
        self.prompt_generator = EquationPromptGenerator(self.model_name)
        self.dataset = self.dataset.map(
            lambda x: self.prompt_generator.generate_prompt(x["nums"], x["target"])
        )
        # Split dataset
        split = self.dataset.train_test_split(test_size=0.1)
        self.train_dataset = split["train"]
        self.test_dataset = split["test"]
        
    def setup_model(self):
        """Initialize the model with M2-compatible configurations"""
        # Check if MPS is available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.backends.mps.is_available() else None,
            trust_remote_code=True,
            load_in_4bit=False,  # Disable 4-bit quantization for M2 compatibility
        )
        
        if not torch.backends.mps.is_available():
            self.model = self.model.to(device)
        
    def setup_training_config(self):
        """Setup PEFT and training configurations optimized for M2"""
        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Adjusted batch sizes and optimization settings for M2
        self.training_args = GRPOConfig(
            output_dir="qwen-r1-aha-moment",
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            logging_steps=10,
            max_steps=1000,
            per_device_train_batch_size=4,  # Reduced batch size
            gradient_accumulation_steps=8,  # Increased gradient accumulation
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=False,  # Disabled mixed precision
            max_prompt_length=128,
            max_completion_length=256,
            num_generations=2,
            beta=0.1,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=100,
        )
        
    def train(self):
        """Initialize trainer and start training with memory optimization"""
        # Enable garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[
                RewardFunctions.format_reward_func,
                RewardFunctions.equation_reward_func
            ],
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            peft_config=self.peft_config,
        )
        
        trainer.train()
        trainer.save_model(self.training_args.output_dir)

def main():
    # Initialize and run training with smaller dataset for testing
    trainer = ModelTrainer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_id="Jiayi-Pan/Countdown-Tasks-3to4",
        num_samples=1000  # Reduced sample size for M2
    )
    trainer.train()

if __name__ == "__main__":
    main()
