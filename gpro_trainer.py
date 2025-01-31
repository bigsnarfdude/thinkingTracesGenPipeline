import os
import sys
import logging
import time
import traceback
import gc
import functools
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationMixin,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from trl import PPOTrainer, PPOConfig
import copy

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging with both file and console output."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, 'training.log'))
        ]
    )
    return logging.getLogger(__name__)

def log_model_loading(func):
    """Decorator to log model loading time and execution status."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info(f"{'='*50}")
        logger.info(f"Starting: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Completed: {func.__name__}")
            logger.info(f"Time taken: {elapsed_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

class MemoryAugmentedRewardModel(PreTrainedModel):
    """GRPO reward model with memory augmentation"""
    def __init__(self, config):
        super().__init__(config)
        self.backbone = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Memory components
        self.positive_memory = []
        self.memory_threshold = 0.8
        self.max_memory_size = 1000

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            
        hidden_states = self.backbone(inputs_embeds)
        values = self.value_head(hidden_states)
        logits = self.reward_head(hidden_states)
        
        # Update memory with high-reward examples
        with torch.no_grad():
            scores = self.get_reward_value(logits)
            high_reward_mask = scores > self.memory_threshold
            if high_reward_mask.any():
                positive_states = hidden_states[high_reward_mask].detach().cpu().numpy()
                self.positive_memory.extend(positive_states)
                if len(self.positive_memory) > self.max_memory_size:
                    self.positive_memory = self.positive_memory[-self.max_memory_size:]
        
        if return_dict:
            return {
                'hidden_states': hidden_states,
                'value': values,
                'logits': logits
            }
        return values, logits

    def get_reward_value(self, logits):
        base_reward = torch.sigmoid(logits).squeeze(-1)
        
        if self.positive_memory:
            memory_similarity = self._compute_memory_similarity(
                self.backbone(logits.detach())
            )
            return 0.7 * base_reward + 0.3 * memory_similarity
        
        return base_reward

    def _compute_memory_similarity(self, features):
        features = features.cpu().numpy()
        if not self.positive_memory:
            return torch.zeros(features.shape[0], device=features.device)
        
        memory_array = np.array(self.positive_memory)
        similarities = np.dot(features, memory_array.T)
        norm_features = np.linalg.norm(features, axis=1, keepdims=True)
        norm_memory = np.linalg.norm(memory_array, axis=1)
        similarities = similarities / (norm_features * norm_memory)
        
        return torch.tensor(similarities.max(axis=1), device=features.device)

class GRPOTrainer:
    """GRPO trainer integrated with TRL's PPOTrainer"""
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        sft_model_path: str = "./qwen-sft-lora-epoch2-final",
        dataset_name: str = "bespokelabs/Bespoke-Stratos-35k",
        output_dir: str = "./grpo-training",
        positive_keywords: List[str] = None,
        negative_keywords: List[str] = None,
    ):
        self.base_model_name = base_model_name
        self.sft_model_path = sft_model_path
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.positive_keywords = positive_keywords or ["helpful", "appropriate", "safe", "good"]
        self.negative_keywords = negative_keywords or ["harmful", "inappropriate", "unsafe", "bad"]

        os.makedirs(output_dir, exist_ok=True)
        self.logger = setup_logging(output_dir)
        
        # Initialize components
        self.tokenizer = self._initialize_tokenizer()
        self.ppo_config = self._create_ppo_config()
        self.train_dataset = self._prepare_dataset()
        self.eval_dataset = self.train_dataset

        # Initialize models
        self.policy_model = None
        self.ref_model = None
        self.reward_model = None

    def _initialize_tokenizer(self):
        """Initialize and configure the tokenizer."""
        try:
            self.logger.info("Initializing tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = '</s>'
            
            tokenizer.padding_side = 'left'
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error initializing tokenizer: {e}")
            raise

    def _create_ppo_config(self) -> PPOConfig:
        """Create PPO training configuration."""
        return PPOConfig(
            learning_rate=5e-4,
            batch_size=20,
            mini_batch_size=10,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            num_train_epochs=1,
            output_dir=self.output_dir,
            response_length=348,
            per_device_train_batch_size=4,
            local_rollout_forward_batch_size=4
        )

    def _prepare_dataset(self):
        """Prepare and format the training dataset."""
        dataset = load_dataset(self.dataset_name)
        dataset = dataset['train'].select(range(5000))

        def format_conversation(example):
            messages = self._parse_conversation(example['conversations'])
            user_msg = [msg.value for msg in messages if msg.from_ == "user"][0]

            inputs = self.tokenizer(
                user_msg,
                truncation=True,
                padding='max_length',
                max_length=512
            )

            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

        return dataset.map(
            format_conversation,
            remove_columns=dataset.column_names
        )

    @staticmethod
    def _parse_conversation(example):
        """Parse conversation data into a standard format."""
        if isinstance(example, list):
            return [
                {'from_': msg.get('from', ''), 'value': msg.get('value', '')}
                for msg in example
            ]
        return [
            {'from_': 'user', 'value': example['input']},
            {'from_': 'assistant', 'value': example['output']}
        ]

    @log_model_loading
    def _initialize_models(self):
        """Initialize all required models for GRPO training."""
        try:
            self.logger.info("Initializing models...")
            
            # Load base model
            self.logger.info("Loading policy model...")
            self.policy_model = self._load_base_model()
            
            # Create reference model
            self.logger.info("Creating reference model...")
            self.ref_model = self._create_reference_model(self.policy_model)
            
            # Initialize reward model
            self.logger.info("Initializing reward model...")
            self.reward_model = self._initialize_reward_model(self.policy_model.config)
            
            return self.policy_model, self.ref_model, self.reward_model
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _load_base_model(self):
        """Load and configure the base model."""
        try:
            self.logger.info(f"Loading base model: {self.base_model_name}")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                compute_dtype=torch.float16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                quantization_config=quantization_config,
                use_cache=False
            )
            
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False
            )
            
            base_model = get_peft_model(base_model, peft_config)
            
            if os.path.exists(self.sft_model_path):
                self.logger.info(f"Loading adapters from {self.sft_model_path}")
                base_model.load_adapter(self.sft_model_path, adapter_name="default")
            
            base_model = prepare_model_for_kbit_training(base_model)
            
            # Set training parameters
            for name, param in base_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    param.data = param.data.to(torch.float32)
            
            base_model.train()
            return base_model
            
        except Exception as e:
            self.logger.error(f"Error loading base model: {str(e)}")
            raise

    def _create_reference_model(self, base_model):
        """Create a reference model from the base model."""
        try:
            ref_model = copy.deepcopy(base_model)
            ref_model.eval()
            return ref_model
        except Exception as e:
            self.logger.error(f"Error creating reference model: {str(e)}")
            raise

    def _initialize_reward_model(self, config):
        """Initialize the reward model."""
        try:
            self.logger.info("Initializing reward model...")
            reward_model = MemoryAugmentedRewardModel(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            reward_model = reward_model.to(device)
            reward_model.train()
            return reward_model
        except Exception as e:
            self.logger.error(f"Error initializing reward model: {str(e)}")
            raise

    def _run_training_loop(self, ppo_trainer):
        """Execute training loop with proper logging and checkpointing."""
        try:
            for epoch in range(self.ppo_config.num_train_epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{self.ppo_config.num_train_epochs}")

                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()

                # Train
                ppo_trainer.train()

                # Save checkpoint
                if self.output_dir:
                    checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch + 1}")
                    self.logger.info(f"Saving checkpoint to {checkpoint_dir}")
                    ppo_trainer.save_model(checkpoint_dir)

            # Save final model
            if self.output_dir:
                final_dir = os.path.join(self.output_dir, "final-model")
                self.logger.info(f"Saving final model to {final_dir}")
                ppo_trainer.save_model(final_dir)

        except Exception as e:
            self.logger.error(f"Training loop failed: {str(e)}")
            raise

    def train(self):
        """Main training method combining GRPO with PPO."""
        try:
            if any(model is None for model in (self.policy_model, self.ref_model, self.reward_model)):
                self._initialize_models()

            if not isinstance(self.reward_model, nn.Module):
                raise ValueError("reward_model must be an instance of nn.Module")

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.policy_model.parameters()),
                lr=self.ppo_config.learning_rate,
                weight_decay=0.01,
                eps=1e-8
            )

            # Initialize PPO trainer with GRPO reward model
            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                policy=self.policy_model,
                ref_policy=self.ref_model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                reward_model=self.reward_model,
                value_model=self.reward_model,
                eval_dataset=self.eval_dataset
            )

            # Run training loop
            self._run_training_loop(ppo_trainer)

            return ppo_trainer

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main execution point for training."""
    try:
        # Initialize trainer
        trainer = GRPOTrainer(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            sft_model_path="./qwen-sft-lora-epoch2-final",
            dataset_name="bespokelabs/Bespoke-Stratos-35k",
            output_dir="./grpo-ppo-training",
            positive_keywords=["helpful", "appropriate", "safe", "good"],
            negative_keywords=["harmful", "inappropriate", "unsafe", "bad"]
        )

        # Start training
        trainer.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
