# binary_memory_cold_start_working_ppo.py

import os
import sys
import logging
import time
import traceback
import gc
import functools
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn

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
from torch.optim import AdamW
import copy


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging with both file and console output.
    Args:
        output_dir: Directory where log file will be saved
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, 'training.log'))
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging("./my_output_dir")
logger.info("Starting process")
logger.warning("Warning message")
logger.error("Error occurred")


def log_model_loading(func):
    """
    Decorator to log model loading time and execution status.
    Args:
        func: Function to be decorated
    Returns:
        Wrapped function with logging
    """
    @functools.wraps(func)  # Preserves metadata of original function
    def wrapper(*args, **kwargs):
        logger.info(f"{'='*50}")
        logger.info(f"Starting: {func.__name__}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            logger.info(f"{'='*50}")
            logger.info(f"Completed: {func.__name__}")
            logger.info(f"Time taken: {elapsed_time:.2f} seconds")
            logger.info(f"{'='*50}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    return wrapper

@dataclass
class ModelOutput:
    """Simple output class to match expected interface."""
    hidden_states: Tuple[torch.Tensor, ...]  # Tuple of tensors
    value: Optional[torch.Tensor] = None     # Optional tensor with default None
    logits: Optional[torch.Tensor] = None    # Optional tensor with default None

class BackboneModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)
        self.config = config
    
    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        """Forward pass of backbone model."""
        try:
            # Handle input embeddings
            if input_ids is not None:
                hidden_states = self.embeddings(input_ids)
            elif hidden_states is None:
                raise ValueError("Either input_ids or hidden_states must be provided")
            
            # Ensure hidden_states is float32
            hidden_states = hidden_states.to(torch.float32)
            
            # Process through layers
            x = self.dense1(hidden_states)
            x = self.act1(x)
            x = self.dropout1(x)
            x = self.dense2(x)
            x = self.act2(x)
            x = self.dropout2(x)
            
            logger.info(f"Output shape: {x.shape}")
            return ModelOutput(hidden_states=(hidden_states, x))
            
        except Exception as e:
            logger.error(f"Error in backbone forward pass: {str(e)}")
            raise


class BinaryRewardModel(PreTrainedModel, GenerationMixin):
    base_model_prefix = "binary_reward_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.binary_reward_model = BackboneModel(config)
        
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

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, return_dict=True, **kwargs):
        outputs = self.binary_reward_model(
            input_ids=input_ids,
            hidden_states=inputs_embeds
        )
        features = outputs.hidden_states[-1]
        
        values = self.value_head(features)
        logits = self.reward_head(features)
        
        if return_dict:
            return ModelOutput(
                hidden_states=outputs.hidden_states,
                value=values,
                logits=logits
            )
        return (values, logits)

    def score(self, hidden_states):
        return self.reward_head(hidden_states)

    def get_reward_value(self, logits):
        scores = torch.sigmoid(logits)
        return scores.squeeze(-1)
    

class MemoryAugmentedRewardModel(BinaryRewardModel):
    def __init__(self, config):
        super().__init__(config)
        self.positive_memory = []
        self.memory_threshold = 0.8
        
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        
        with torch.no_grad():
            if outputs.logits is not None:
                scores = self.get_reward_value(outputs.logits)
                high_reward_mask = scores > self.memory_threshold
                
                if high_reward_mask.any():
                    positive_states = outputs.hidden_states[-1][high_reward_mask]
                    self.positive_memory.extend(positive_states.cpu().numpy())
                    if len(self.positive_memory) > 1000:
                        self.positive_memory = self.positive_memory[-1000:]
        
        return outputs
    
    def score(self, hidden_states):
        base_score = super().score(hidden_states)
        
        if self.positive_memory:
            memory_similarity = self._compute_memory_similarity(hidden_states)
            return 0.7 * base_score + 0.3 * memory_similarity
            
        return base_score
        
    def get_reward_value(self, logits):
        base_reward = super().get_reward_value(logits)
        
        if self.positive_memory and logits is not None:
            features = self.binary_reward_model(logits)[-1]
            memory_similarity = self._compute_memory_similarity(features)
            enhanced_reward = 0.7 * base_reward + 0.3 * memory_similarity
            return enhanced_reward
            
        return base_reward
    
    def _compute_memory_similarity(self, features):
        features = features.cpu().numpy()
        similarities = np.dot(features, np.array(self.positive_memory).T)
        similarities = similarities / (np.linalg.norm(features) * np.linalg.norm(self.positive_memory, axis=1))
        return torch.tensor(similarities.max(axis=1), device=features.device)



def get_reward(reward_model, sequences, pad_token_id, context_length):
    """Get reward for a batch of sequences."""
    reward_model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = reward_model(sequences)
        
        # Get sequence length and verify context length
        seq_length = sequences.size(1)
        safe_context_length = min(context_length, seq_length - 1)
        
        # Extract rewards and values
        reward_logits = outputs.logits[:, safe_context_length:-1]
        rewards = reward_model.get_reward_value(reward_logits)
        values = outputs.value[:, safe_context_length:-1] if outputs.value is not None else None
        
        return values, rewards, outputs.hidden_states


@dataclass
class Message:
    from_: str
    value: str

def parse_conversation(example):
    if isinstance(example, list):
        return [Message(from_=msg.get('from', ''), value=msg.get('value', '')) for msg in example]
    return [Message(from_='user', value=example['input']), Message(from_='assistant', value=example['output'])]


class QwenPEFTTrainer:
    """Trainer class for fine-tuning Qwen models using PPO."""
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
        sft_model_path: str = "./qwen-sft-lora-epoch2-final",
        dataset_name: str = "bespokelabs/Bespoke-Stratos-35k",
        output_dir: str = "./qwen-continued-training",
        positive_keywords: List[str] = None,
        negative_keywords: List[str] = None,
    ):
        self.base_model_name = base_model_name
        self.sft_model_path = sft_model_path 
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.positive_keywords = positive_keywords or ["helpful", "appropriate", "safe", "good"]
        self.negative_keywords = negative_keywords or ["harmful", "inappropriate", "unsafe", "bad"]

        # Set up directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self._setup_logging()
        self.tokenizer = self._initialize_tokenizer()
        self.ppo_config = self._create_ppo_config()
        self.train_dataset = self._prepare_dataset()
        self.eval_dataset = self.train_dataset
        self.dataset = self.train_dataset

        # Initialize models
        self.policy_model = None
        self.ref_model = None 
        self.reward_model = None


    def _setup_logging(self):
        """Configure logging with both file and console output."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.output_dir, 'training.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_ppo_config(self) -> PPOConfig:
        """Create PPO training configuration with appropriate batch sizes."""
        return PPOConfig(
            learning_rate=1e-4,
            batch_size=8,  # Increased from 1
            mini_batch_size=4,  # Increased from 1
            gradient_accumulation_steps=4,  # Increased from 1
            max_grad_norm=1.0,
            num_train_epochs=1,
            output_dir=self.output_dir,
            response_length=128,
            per_device_train_batch_size=2,  # Increased from 1
            local_rollout_forward_batch_size=2  # Increased from 1
        )

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

    def _prepare_dataset(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset['train'].select(range(5000))

        def format_conversation(example):
            messages = parse_conversation(example['conversations'])
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


    @log_model_loading
    def _initialize_models(self):
        """Initialize all required models for PPO training."""
        try:
            self.logger.info("Initializing models...")
           
            self.logger.info("Loading base model...")
            self.policy_model = self._load_base_model()
           
            self.logger.info("Creating reference model...")
            self.ref_model = self._create_reference_model(self.policy_model)
           
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
        """Initialize the reward model with the given config."""
        try:
            self.logger.info("Initializing reward model...")
            reward_config = PretrainedConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_hidden_layers=4
            )
            reward_model = MemoryAugmentedRewardModel(reward_config)
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

                torch.cuda.empty_cache()
                gc.collect()

                ppo_trainer.train()

                if self.output_dir:
                    checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch + 1}")
                    self.logger.info(f"Saving checkpoint to {checkpoint_dir}")
                    ppo_trainer.save_model(checkpoint_dir)

            if self.output_dir:
                final_dir = os.path.join(self.output_dir, "final-model")
                self.logger.info(f"Saving final model to {final_dir}")
                ppo_trainer.save_model(final_dir)

        except Exception as e:
            self.logger.error(f"Training loop failed: {str(e)}")
            raise

    def train(self):
        """Main training method."""
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

            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                policy=self.policy_model,
                ref_policy=self.ref_model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                reward_model=self.reward_model,
                value_model=self.reward_model,
                eval_dataset=self.dataset
            )

            self._run_training_loop(ppo_trainer)

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main execution point for training."""
    try:
        trainer = QwenPEFTTrainer(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            sft_model_path="./qwen-sft-lora-epoch2-final",
            dataset_name="bespokelabs/Bespoke-Stratos-35k",
            output_dir="./qwen-ppo-training",
            positive_keywords=["helpful", "appropriate", "safe", "good"],
            negative_keywords=["harmful", "inappropriate", "unsafe", "bad"]
        )

        trainer.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
