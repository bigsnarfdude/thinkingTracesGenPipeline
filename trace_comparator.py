import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class TraceComparator:
    def __init__(self, llm_evaluator):
        self.evaluator = llm_evaluator
        self.comparison_results = defaultdict(list)
    
    def load_data(self, human_traces_path: str, generated_traces_path: str) -> tuple:
        """Load both human and generated traces"""
        human_traces = {}
        with open(human_traces_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                human_traces[data['input']] = data['thinking_trace']
                
        generated_traces = {}
        with open(generated_traces_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                generated_traces[data['input']] = data['generated_trace']
                
        # Find common questions
        common_questions = set(human_traces.keys()) & set(generated_traces.keys())
        print(f"Found {len(common_questions)} questions with both human and generated traces")
        
        return human_traces, generated_traces, common_questions

    def compare_traces(self, human_traces: Dict, generated_traces: Dict, 
                      common_questions: set, sample_size: int = None):
        """Compare human vs generated traces for common questions"""
        # Sample questions if specified
        questions = list(common_questions)
        if sample_size:
            questions = questions[:sample_size]
            
        print(f"Comparing traces for {len(questions)} questions...")
        
        for question in questions:
            human_trace = human_traces[question]
            generated_trace = generated_traces[question]
            
            # Get detailed evaluation
            comparison = self.evaluator.comprehensive_evaluation(
                generated_trace, reference=human_trace
            )
            
            # Store results
            self.comparison_results['question'].append(question)
            self.comparison_results['human_trace'].append(human_trace)
            self.comparison_results['generated_trace'].append(generated_trace)
            self.comparison_results['reasoning_depth'].append(
                comparison['reasoning_depth']['score'])
            self.comparison_results['step_coherence'].append(
                comparison['step_coherence']['score'])
            self.comparison_results['verification_quality'].append(
                comparison['verification_quality']['score'])
            self.comparison_results['completeness'].append(
                comparison['completeness']['score'])
            self.comparison_results['overall_score'].append(
                comparison['overall_score'])
            # Store detailed analysis
            self.comparison_results['detailed_analysis'].append(comparison)

    def analyze_results(self, output_dir: Path):
        """Analyze comparison results and generate visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame({
            'Reasoning Depth': self.comparison_results['reasoning_depth'],
            'Step Coherence': self.comparison_results['step_coherence'],
            'Verification': self.comparison_results['verification_quality'],
            'Completeness': self.comparison_results['completeness'],
            'Overall Score': self.comparison_results['overall_score']
        })
        
        # Generate visualizations
        # 1. Score Distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df)
        plt.title('Distribution of Quality Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png')
        plt.close()
        
        # 2. Score Correlations
        plt.figure(figsize=(8, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Score Correlations')
        plt.tight_layout()
        plt.savefig(output_dir / 'score_correlations.png')
        plt.close()
        
        # Generate summary statistics
        summary = {
            'mean_scores': df.mean().to_dict(),
            'median_scores': df.median().to_dict(),
            'std_scores': df.std().to_dict(),
            'num_compared': len(df),
            'high_quality_ratio': (df['Overall Score'] > 0.8).mean()
        }
        
        # Save detailed results
        with open(output_dir / 'comparison_results.jsonl', 'w') as f:
            for i in range(len(self.comparison_results['question'])):
                result = {
                    'question': self.comparison_results['question'][i],
                    'human_trace': self.comparison_results['human_trace'][i],
                    'generated_trace': self.comparison_results['generated_trace'][i],
                    'scores': {
                        'reasoning_depth': self.comparison_results['reasoning_depth'][i],
                        'step_coherence': self.comparison_results['step_coherence'][i],
                        'verification_quality': self.comparison_results['verification_quality'][i],
                        'completeness': self.comparison_results['completeness'][i],
                        'overall_score': self.comparison_results['overall_score'][i]
                    },
                    'detailed_analysis': self.comparison_results['detailed_analysis'][i]
                }
                f.write(json.dumps(result) + '\n')
        
        # Save summary
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

if __name__ == "__main__":
    from llm_evaluator import LLMEvaluator
    
    # Initialize evaluator and comparator
    evaluator = LLMEvaluator()
    comparator = TraceComparator(evaluator)
    
    # Load and compare traces
    human_traces, generated_traces, common_qs = comparator.load_data(
        "data/human_traces.jsonl",
        "data/generated_traces.jsonl"
    )
    
    # Compare traces (use sample_size for testing)
    comparator.compare_traces(human_traces, generated_traces, common_qs, sample_size=100)
    
    # Analyze and save results
    summary = comparator.analyze_results("results/trace_comparison")
    print("\nSummary Statistics:")
    for metric, value in summary['mean_scores'].items():
        print(f"Average {metric}: {value:.3f}")
    print(f"\nHigh Quality Ratio: {summary['high_quality_ratio']:.2%}")
