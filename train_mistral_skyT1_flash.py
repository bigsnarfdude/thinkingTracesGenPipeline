import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import gc
from torch.cuda import empty_cache
import wandb
import os
from datetime import datetime

class RLTrainer:
    def __init__(
        self, 
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        project_name="rl-instruct-training",
        load_in_8bit=True
    ):
        # Initialize wandb
        self.run = wandb.init(
            project=project_name,
            config={
                "model_name": model_name,
                "load_in_8bit": load_in_8bit,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        )
        
        print(f"Loading model {model_name} with 8-bit quantization...")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        self.instruction_template = "<s>[INST] {instruction} [/INST]"
        
        # Log model info to wandb
        wandb.config.update({
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "instruction_template": self.instruction_template
        })

    def prepare_batch(self, examples, batch_size=1):
        """Prepare a small batch of data with memory efficiency."""
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            prompts = [
                self.instruction_template.format(
                    instruction=f"Solve this problem step by step:\n{ex['prompt']}"
                ) for ex in batch
            ]
            
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            yield {
                'inputs': inputs,
                'chosen': [ex['chosen'] for ex in batch],
                'rejected': [ex['rejected'] for ex in batch]
            }

    def compute_reward(self, output, chosen, rejected):
        """Compute reward with metrics logging."""
        output_tokens = set(output.split())
        chosen_tokens = set(chosen.split())
        rejected_tokens = set(rejected.split())
        
        chosen_sim = len(output_tokens & chosen_tokens) / len(output_tokens | chosen_tokens)
        rejected_sim = len(output_tokens & rejected_tokens) / len(output_tokens | rejected_tokens)
        
        # Log similarities to wandb
        wandb.log({
            "chosen_similarity": chosen_sim,
            "rejected_similarity": rejected_sim
        })
        
        return chosen_sim - rejected_sim

    def train_step(self, batch, optimizer, global_step):
        """Training step with detailed monitoring."""
        try:
            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **batch['inputs'],
                    max_length=1024,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7
                )
                generated_texts = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
            
            # Compute rewards
            rewards = [
                self.compute_reward(gen, chosen, rejected)
                for gen, chosen, rejected in zip(
                    generated_texts, 
                    batch['chosen'],
                    batch['rejected']
                )
            ]
            rewards = torch.tensor(rewards, device='cuda')
            
            # Forward pass
            outputs = self.model(**batch['inputs'])
            log_probs = outputs.logits.log_softmax(-1)
            
            # Compute losses
            policy_loss = -log_probs.mean() * rewards.mean()
            
            # Backward pass
            optimizer.zero_grad()
            policy_loss.backward()
            
            # Get gradient norm for monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            # Log metrics
            metrics = {
                "policy_loss": policy_loss.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "gradient_norm": grad_norm.item(),
                "global_step": global_step
            }
            wandb.log(metrics)
            
            # Log sample outputs periodically
            if global_step % 100 == 0:
                wandb.log({
                    "sample_output": wandb.Table(
                        columns=["Generated", "Chosen", "Rejected"],
                        data=[[generated_texts[0], batch['chosen'][0], batch['rejected'][0]]]
                    )
                })
            
            # Clear cache
            del outputs, log_probs
            empty_cache()
            
            return policy_loss.item()
            
        except RuntimeError as e:
            print(f"Error in training step: {e}")
            wandb.log({"training_errors": str(e)})
            empty_cache()
            return None

    def train(
        self,
        num_epochs=3,
        batch_size=1,
        learning_rate=1e-5,
        max_samples=1000
    ):
        """Training loop with comprehensive monitoring."""
        print("Loading dataset...")
        dataset = load_dataset(
            "NovaSky-AI/Sky-T1_preference_data_10k",
            split=f'train[:{max_samples}]'
        )
        
        wandb.config.update({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "dataset_size": len(dataset)
        })
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        global_step = 0
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            progress_bar = tqdm(
                self.prepare_batch(dataset, batch_size),
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch in progress_bar:
                loss = self.train_step(batch, optimizer, global_step)
                global_step += 1
                
                if loss is not None:
                    total_loss += loss
                    batch_count += 1
                    progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                
                if batch_count % 10 == 0:
                    empty_cache()
                    gc.collect()
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            wandb.log({
                "epoch": epoch + 1,
                "average_epoch_loss": avg_loss
            })
            
            # Save checkpoint with wandb
            checkpoint_path = f"checkpoint_epoch_{epoch+1}"
            self.save_checkpoint(checkpoint_path)
            
            empty_cache()
            gc.collect()

    def save_checkpoint(self, path):
        """Save checkpoint and log to wandb."""
        print(f"Saving checkpoint to {path}...")
        self.model.save_pretrained(
            path,
            safe_serialization=True,
            max_shard_size="500MB"
        )
        self.tokenizer.save_pretrained(path)
        
        # Log checkpoint to wandb
        wandb.save(f"{path}/*")

def main():
    # Set up wandb login
    wandb.login()
    
    # Initialize trainer
    trainer = RLTrainer(
        project_name="rl-instruct-training",
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # Train model
    trainer.train(
        num_epochs=3,
        batch_size=1,
        learning_rate=1e-5,
        max_samples=1000
    )
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
