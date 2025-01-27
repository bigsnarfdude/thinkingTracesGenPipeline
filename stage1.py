import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import json
from pathlib import Path
from tqdm import tqdm
import gc

class ReasoningTraceGenerator:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", 
                 device="cuda", load_in_8bit=False):
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        print(f"Loading tokenizer from {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print(f"Loading base model {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add special tokens if not present
        special_tokens = ["<|begin_of_thought|>", "<|end_of_thought|>",
                         "<|begin_of_solution|>", "<|end_of_solution|>"]
        if not all(token in self.tokenizer.get_vocab() for token in special_tokens):
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load_lora_weights(self, lora_path):
        """Load LoRA weights if available"""
        if Path(lora_path).exists():
            print(f"Loading LoRA weights from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
    
    def generate_trace(self, input_text, max_length=2048):
        """Generate reasoning trace for single input"""
        prompt = f"Given this input, think through the solution step by step.\n\nInput: {input_text}\n\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Remove prompt from response
        response = response[len(prompt):]
        return response

    def process_dataset(self, input_jsonl, output_jsonl, batch_size=1):
        """Process entire dataset and save reasoning traces"""
        # Create output directory if needed
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input data
        with open(input_jsonl, 'r') as f:
            data = [json.loads(line) for line in f]
        
        print(f"Processing {len(data)} examples...")
        with open(output_jsonl, 'w') as f:
            for i in tqdm(range(0, len(data), batch_size)):
                batch = data[i:i + batch_size]
                
                for item in batch:
                    input_text = item['input']
                    trace = self.generate_trace(input_text)
                    
                    # Save original input and generated trace
                    output = {
                        'input': input_text,
                        'generated_trace': trace
                    }
                    f.write(json.dumps(output) + '\n')
                
                # Clear cache periodically
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

    def cleanup(self):
        """Free up memory"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Load config
    from config import *
    
    # Initialize generator
    generator = ReasoningTraceGenerator(
        base_model_name=MODEL_NAME,
        load_in_8bit=True  # Use 8-bit quantization for memory efficiency
    )
    
    # Load LoRA weights if available (epoch1)
    generator.load_lora_weights(EPOCH1_MODEL_PATH)
    
    # Process dataset
    generator.process_dataset(
        input_jsonl=RAW_DATA_PATH,
        output_jsonl=GENERATED_RESPONSES_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Cleanup
    generator.cleanup()
