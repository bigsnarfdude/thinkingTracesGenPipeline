import os
import sys
import logging
import time
import traceback
import gc
import functools
from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

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

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    num_samples: int = 64
    learning_rate: float = 1e-6
    kl_coef: float = 0.04
    max_length: int = 1024
    batch_size: int = 1024
    output_dir: str = "./grpo_output"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

class BackboneRewardModel(nn.Module):
    """Backbone network for reward computation"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.config = config

    def forward(self, hidden_states):
        return self.dense(hidden_states)

class MemoryAugmentedRewardModel(PreTrainedModel):
    """Reward model with memory augmentation"""
    def __init__(self, config):
        super().__init__(config)
        self.backbone = BackboneRewardModel(config)
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
        self.positive_memory = []
        self.memory_threshold = 0.8

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        hidden_states = self.backbone(inputs_embeds if inputs_embeds is not None else input_ids)
        
        values = self.value_head(hidden_states)
        logits = self.reward_head(hidden_states)
        
        with torch.no_grad():
            scores = self.get_reward_value(logits)
            high_reward_mask = scores > self.memory_threshold
            if high_reward_mask.any():
                positive_states = hidden_states[high_reward_mask]
                self.positive_memory.extend(positive_states.cpu().numpy())
                if len(self.positive_memory) > 1000:
                    self.positive_memory = self.positive_memory[-1000:]
        
        return {
            'hidden_states': hidden_states,
            'value': values,
            'logits': logits
        }

    def get_reward_value(self, logits):
        base_reward = torch.sigmoid(logits).squeeze(-1)
        if self.positive_memory and logits is not None:
            memory_similarity = self._compute_memory_similarity(
                self.backbone(logits)
            )
            return 0.7 * base_reward + 0.3 * memory_similarity
        return base_reward

    def _compute_memory_similarity(self, features):
        features = features.cpu().numpy()
        if not self.positive_memory:
            return torch.zeros(features.shape[0], device=features.device)
        similarities = np.dot(features, np.array(self.positive_memory).T)
        norm_features = np.linalg.norm(features, axis=1, keepdims=True)
        norm_memory = np.linalg.norm(self.positive_memory, axis=1)
        similarities = similarities / (norm_features * norm_memory)
        return torch.tensor(similarities.max(axis=1), device=features.device)

class GRPO:
    """Group Relative Policy Optimization implementation"""
    def __init__(
        self,
        config: GRPOConfig,
        policy_model: PreTrainedModel,
        reward_model: MemoryAugmentedRewardModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.logger = setup_logging(config.output_dir)
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )

    @log_model_loading
    def sample_outputs(self, question: str) -> List[str]:
        """Generate multiple outputs for a question"""
        outputs = []
        encoded_question = self.tokenizer(
            question,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.policy_model.device)

        for _ in range(self.config.num_samples):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    **encoded_question,
                    max_length=self.config.max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.7
                )
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                outputs.append(output_text)
        return outputs

    def compute_rewards(self, question: str, outputs: List[str]) -> torch.Tensor:
        """Compute rewards for a group of outputs"""
        inputs = [f"Question: {question}\nAnswer: {output}" for output in outputs]
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.reward_model.device)

        with torch.no_grad():
            reward_outputs = self.reward_model(**encoded)
            rewards = self.reward_model.get_reward_value(reward_outputs['logits'])
        return rewards

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards within the group"""
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward == 0:
            return torch.zeros_like(rewards)
        return (rewards - mean_reward) / (std_reward + 1e-8)

    def train_step(self, question: str) -> Tuple[float, float]:
        """Execute one GRPO training step"""
        try:
            # Generate group of outputs
            outputs = self.sample_outputs(question)
            
            # Compute and normalize rewards
            rewards = self.compute_rewards(question, outputs)
            advantages = self.normalize_rewards(rewards)
            
            total_loss = 0
            policy_loss = 0
            kl_loss = 0
            
            # Update policy for each output
            for output, advantage in zip(outputs, advantages):
                # Get model outputs
                inputs = self.tokenizer(
                    f"Question: {question}\nAnswer: {output}",
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.policy_model.device)
                
                model_outputs = self.policy_model(**inputs)
                logits = model_outputs.logits
                
                # Compute losses
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pg_loss = -advantage * log_probs.mean()
                
                # Add KL penalty
                with torch.no_grad():
                    ref_logits = self.policy_model(**inputs).logits
                kl_div = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits, dim=-1),
                    torch.nn.functional.softmax(ref_logits, dim=-1),
                    reduction='batchmean'
                )
                
                # Combined loss
                loss = pg_loss + self.config.kl_coef * kl_div
                
                # Accumulate losses
                policy_loss += pg_loss.item()
                kl_loss += kl_div.item()
                total_loss += loss

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return policy_loss / len(outputs), kl_loss / len(outputs)
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        config = GRPOConfig()
        logger = setup_logging(config.output_dir)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Initialize policy model
        policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Add LoRA adaptation
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        policy_model = get_peft_model(policy_model, peft_config)
        
        # Initialize reward model
        reward_model = MemoryAugmentedRewardModel(policy_model.config)
        
        # Initialize GRPO trainer
        grpo_trainer = GRPO(
            config=config,
            policy_model=policy_model,
            reward_model=reward_model,
            tokenizer=tokenizer
        )
        
        # Training loop
        questions = ["Sample question 1", "Sample question 2"]  # Replace with actual dataset
        for epoch in range(3):
            logger.info(f"Starting epoch {epoch + 1}")
            for question in questions:
                policy_loss, kl_loss = grpo_trainer.train_step(question)
                logger.info(f"Policy loss: {policy_loss:.4f}, KL loss: {kl_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{epoch + 1}")
            policy_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
