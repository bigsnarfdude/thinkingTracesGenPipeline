import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

class RLTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
        self.device = device
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
            device_map="auto"  # Automatically handle model parallelism
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_training_data(self, dataset):
        """Format the HuggingFace dataset for training."""
        def format_prompt(example):
            # Format the conversation into a prompt
            return {
                'prompt': f"Human: {example['prompt']}\nAssistant:",
                'chosen': example['chosen'],
                'rejected': example['rejected']
            }
        
        formatted_data = dataset.map(format_prompt)
        return formatted_data

    def compute_rewards(self, outputs, chosen_outputs, rejected_outputs):
        """
        Compute rewards using preference learning.
        Rewards are higher when outputs are more similar to chosen than rejected responses.
        """
        rewards = []
        
        for output, chosen, rejected in zip(outputs, chosen_outputs, rejected_outputs):
            # Convert to token sets for comparison
            output_tokens = set(output.split())
            chosen_tokens = set(chosen.split())
            rejected_tokens = set(rejected.split())
            
            # Compute similarities
            chosen_similarity = len(output_tokens.intersection(chosen_tokens)) / max(len(output_tokens), len(chosen_tokens))
            rejected_similarity = len(output_tokens.intersection(rejected_tokens)) / max(len(output_tokens), len(rejected_tokens))
            
            # Reward is the difference in similarities (normalized to [-1, 1])
            reward = 2 * (chosen_similarity - rejected_similarity)
            rewards.append(reward)
            
        return torch.tensor(rewards, device=self.device)

    def train_step(self, batch, optimizer, ppo_epochs=4):
        """Perform one training step using PPO."""
        prompts = batch['prompt']
        chosen_outputs = batch['chosen']
        rejected_outputs = batch['rejected']
        
        # Tokenize inputs
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate initial outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7
            )
            initial_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # PPO training loop
        total_loss = 0
        for _ in range(ppo_epochs):
            # Get model outputs
            outputs = self.model(**inputs).logits
            
            # Compute rewards
            rewards = self.compute_rewards(initial_outputs, chosen_outputs, rejected_outputs)
            
            # Compute PPO loss
            log_probs = outputs.log_softmax(-1)
            policy_loss = -log_probs * rewards.unsqueeze(-1).unsqueeze(-1)
            
            # Add value loss (simple L2 loss to predicted values)
            value_pred = outputs.mean(-1)
            value_loss = ((value_pred - rewards.unsqueeze(-1)) ** 2).mean()
            
            # Combine losses
            loss = policy_loss.mean() + 0.5 * value_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / ppo_epochs

    def train(self, num_epochs=3, batch_size=4, learning_rate=1e-5):
        """Train the model using PPO on the HuggingFace dataset."""
        print("Loading dataset...")
        dataset = load_dataset("NovaSky-AI/Sky-T1_preference_data_10k")
        train_dataset = self.prepare_training_data(dataset['train'])
        
        print("Starting training...")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i in progress_bar:
                batch = train_dataset[i:i+batch_size]
                loss = self.train_step(batch, optimizer)
                total_loss += loss
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                
            avg_loss = total_loss / (len(train_dataset) // batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch+1}"
            self.save_model(checkpoint_path)

    def save_model(self, output_dir):
        """Save the model and tokenizer."""
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = RLTrainer()
    
    # Train the model
    trainer.train(
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-5
    )
    
    # Save the final model
    trainer.save_model("fine_tuned_model")
