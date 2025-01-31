import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    num_samples: int = 64  # Number of samples per question
    learning_rate: float = 1e-6
    kl_coef: float = 0.04
    max_length: int = 1024
    batch_size: int = 1024

class RewardModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel):
        """Initialize reward model using a pre-trained model as base"""
        super().__init__()
        self.base_model = base_model
        self.score_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward scores for inputs
        Args:
            input_ids: Tensor of token ids
            attention_mask: Attention mask for input tokens
        Returns:
            Tensor of reward scores
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # Use [CLS] token representation for scoring
        scores = self.score_head(last_hidden[:, 0, :])
        return scores

class GRPO:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: GRPOConfig
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )

    def sample_outputs(self, question: str) -> List[str]:
        """Generate multiple outputs for a single question"""
        outputs = []
        for _ in range(self.config.num_samples):
            with torch.no_grad():
                output_ids = self.policy_model.generate(
                    input_ids=self.tokenizer.encode(question, return_tensors='pt'),
                    max_length=self.config.max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.7
                )
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                outputs.append(output_text)
        return outputs

    def compute_rewards(self, question: str, outputs: List[str]) -> torch.Tensor:
        """Compute rewards for a group of outputs"""
        # Combine question with each output for reward computation
        inputs = [f"Question: {question}\nAnswer: {output}" for output in outputs]
        
        # Tokenize all inputs
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Compute rewards using reward model
        with torch.no_grad():
            rewards = self.reward_model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        return rewards.squeeze()

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards within a group"""
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward == 0:
            return torch.zeros_like(rewards)
        return (rewards - mean_reward) / (std_reward + 1e-8)

    def compute_kl_divergence(self, question: str, output: str) -> torch.Tensor:
        """Compute KL divergence between current policy and reference model"""
        inputs = self.tokenizer(
            f"Question: {question}\nAnswer: {output}",
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Get logits from current policy
        policy_logits = self.policy_model(**inputs).logits
        
        # Get logits from reference model (using initial policy model state)
        with torch.no_grad():
            ref_logits = self.policy_model(**inputs).logits
        
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div

    def train_step(self, question: str) -> Tuple[float, float]:
        """Perform one GRPO training step"""
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
            # Compute policy gradient loss
            inputs = self.tokenizer(
                f"Question: {question}\nAnswer: {output}",
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            
            logits = self.policy_model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Policy gradient loss
            pg_loss = -advantage * log_probs.mean()
            
            # KL divergence loss
            kl_div = self.compute_kl_divergence(question, output)
            
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

def train_grpo(
    train_questions: List[str],
    policy_model: PreTrainedModel,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizer,
    config: GRPOConfig,
    num_epochs: int = 1
) -> List[Tuple[float, float]]:
    """
    Train policy model using GRPO
    Returns:
        List of (policy_loss, kl_loss) tuples for each step
    """
    grpo = GRPO(policy_model, reward_model, tokenizer, config)
    training_stats = []
    
    for epoch in range(num_epochs):
        for question in train_questions:
            policy_loss, kl_loss = grpo.train_step(question)
            training_stats.append((policy_loss, kl_loss))
            
    return training_stats

# Example usage:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize models and tokenizer
policy_model = AutoModelForCausalLM.from_pretrained("path/to/model")
base_model = AutoModelForCausalLM.from_pretrained("path/to/model")
reward_model = RewardModel(base_model)
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# Configure GRPO
config = GRPOConfig(
    num_samples=64,
    learning_rate=1e-6,
    kl_coef=0.04
)

# Training data
train_questions = [
    "What is 5 + 7?",
    "Solve: 3x + 2 = 11",
    # ... more questions
]

# Train
training_stats = train_grpo(
    train_questions=train_questions,
    policy_model=policy_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    config=config,
    num_epochs=1
)
"""
