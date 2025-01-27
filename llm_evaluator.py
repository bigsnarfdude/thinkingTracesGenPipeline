from typing import Dict, List
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Evaluation criteria prompts
        self.criteria_prompts = {
            "reasoning_depth": """
Evaluate the depth and sophistication of this reasoning trace. Consider:
1. Logical progression of ideas
2. Use of domain-specific knowledge
3. Consideration of edge cases
4. Level of detail in explanations
5. Connections between concepts

Rate from 1-10 and explain why.

Reasoning trace:
{text}

Analysis:""",
            
            "step_coherence": """
Analyze the coherence between reasoning steps. Consider:
1. Logical flow between steps
2. Clear connections between ideas
3. No missing logical jumps
4. Proper sequencing
5. Clear dependencies between steps

Rate from 1-10 and explain why.

Reasoning trace:
{text}

Analysis:""",
            
            "verification_quality": """
Evaluate the quality of verification in this reasoning. Consider:
1. Presence of explicit verification steps
2. Thoroughness of checking
3. Consideration of assumptions
4. Testing of conclusions
5. Validation against requirements

Rate from 1-10 and explain why.

Reasoning trace:
{text}

Analysis:""",
            
            "completeness": """
Assess how complete and comprehensive the reasoning is. Consider:
1. All aspects of problem addressed
2. No missing steps
3. Proper problem setup
4. Clear final conclusion
5. Handling of all cases

Rate from 1-10 and explain why.

Reasoning trace:
{text}

Analysis:"""
        }
    
    def evaluate_criterion(self, text: str, criterion: str) -> Dict:
        """Evaluate a single criterion using LLM"""
        prompt = self.criteria_prompts[criterion].format(text=text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate analysis
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.3,
                num_return_sequences=1
            )
        
        analysis = self.tokenizer.decode(outputs[0])
        
        # Extract score (1-10) and explanation
        try:
            # Simple extraction - could be made more robust
            score_line = [line for line in analysis.split('\n') 
                         if any(str(i) for i in range(10)) in line][0]
            score = float([num for num in score_line.split() 
                         if num.isdigit()][0])
            
            # Normalize to 0-1
            score = score / 10.0
            
        except:
            score = 0.0
            
        return {
            'score': score,
            'analysis': analysis
        }
    
    def evaluate_against_reference(self, generated: str, reference: str) -> Dict:
        """Compare generated reasoning against reference"""
        prompt = f"""
Compare these two reasoning traces and evaluate how well the generated one matches
the quality and approach of the reference. Consider:
1. Coverage of key points
2. Reasoning approach
3. Level of detail
4. Logic structure
5. Verification methods

Reference trace:
{reference}

Generated trace:
{generated}

Rate similarity from 1-10 and explain why:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.3
            )
        
        analysis = self.tokenizer.decode(outputs[0])
        
        try:
            score_line = [line for line in analysis.split('\n') 
                         if any(str(i) for i in range(10)) in line][0]
            score = float([num for num in score_line.split() 
                         if num.isdigit()][0]) / 10.0
        except:
            score = 0.0
            
        return {
            'similarity_score': score,
            'comparison_analysis': analysis
        }
    
    def comprehensive_evaluation(self, text: str, reference: str = None) -> Dict:
        """Perform comprehensive evaluation of reasoning trace"""
        results = {}
        
        # Evaluate each criterion
        for criterion in self.criteria_prompts.keys():
            results[criterion] = self.evaluate_criterion(text, criterion)
        
        # Compare against reference if provided
        if reference:
            results['reference_comparison'] = self.evaluate_against_reference(
                text, reference
            )
        
        # Calculate overall score
        criterion_scores = [results[c]['score'] for c in self.criteria_prompts.keys()]
        overall_score = sum(criterion_scores) / len(criterion_scores)
        
        if reference:
            overall_score = (overall_score * 0.7 + 
                           results['reference_comparison']['similarity_score'] * 0.3)
        
        results['overall_score'] = overall_score
        
        return results

    def batch_evaluate(self, data_path: str, output_path: str):
        """Evaluate a batch of reasoning traces"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        results = []
        for item in data:
            evaluation = self.comprehensive_evaluation(
                item['generated_trace'],
                item.get('reference_trace')
            )
            
            results.append({
                'input': item['input'],
                'generated_trace': item['generated_trace'],
                'evaluation': evaluation
            })
        
        # Save results
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
