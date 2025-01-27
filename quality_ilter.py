import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

class QualityFilter:
    def __init__(self, thresholds: Dict[str, float] = None):
        # Default quality thresholds
        self.thresholds = thresholds or {
            'reasoning_depth': 0.7,
            'step_coherence': 0.7,
            'verification_quality': 0.6,
            'completeness': 0.7,
            'overall_score': 0.75
        }
        
    def load_comparison_results(self, results_path: str) -> List[Dict]:
        """Load comparison results from JSONL file"""
        results = []
        with open(results_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results

    def filter_high_quality(self, results: List[Dict], 
                          strict: bool = False) -> List[Dict]:
        """Filter for high-quality examples based on thresholds"""
        filtered = []
        
        for result in results:
            scores = result['scores']
            
            if strict:
                # All metrics must meet thresholds
                meets_threshold = all(
                    scores[metric] >= self.thresholds[metric]
                    for metric in self.thresholds.keys()
                )
            else:
                # Overall score and at least 2 other metrics must meet thresholds
                metric_counts = sum(
                    scores[metric] >= self.thresholds[metric]
                    for metric in self.thresholds.keys()
                    if metric != 'overall_score'
                )
                meets_threshold = (
                    scores['overall_score'] >= self.thresholds['overall_score']
                    and metric_counts >= 2
                )
            
            if meets_threshold:
                filtered.append(result)
        
        return filtered

    def create_training_data(self, filtered_results: List[Dict], 
                           output_path: str):
        """Convert filtered results to training data format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for result in filtered_results:
                training_example = {
                    'input': result['question'],
                    'reasoning_trace': result['generated_trace'],
                    'quality_metrics': result['scores']
                }
                f.write(json.dumps(training_example) + '\n')

    def analyze_filtering(self, original: List[Dict], 
                        filtered: List[Dict]) -> Dict:
        """Analyze filtering results"""
        total = len(original)
        kept = len(filtered)
        
        # Calculate average scores before/after
        def avg_scores(results):
            scores_df = pd.DataFrame([r['scores'] for r in results])
            return scores_df.mean().to_dict()
        
        original_scores = avg_scores(original)
        filtered_scores = avg_scores(filtered)
        
        return {
            'total_examples': total,
            'kept_examples': kept,
            'retention_rate': kept / total,
            'original_avg_scores': original_scores,
            'filtered_avg_scores': filtered_scores,
            'score_improvement': {
                k: filtered_scores[k] - original_scores[k]
                for k in original_scores.keys()
            }
        }

    def run_filtering_pipeline(self, 
                             comparison_results_path: str,
                             output_path: str,
                             strict: bool = False) -> Dict:
        """Run complete filtering pipeline"""
        # Load results
        results = self.load_comparison_results(comparison_results_path)
        
        # Filter high-quality examples
        filtered = self.filter_high_quality(results, strict=strict)
        
        # Create training data
        self.create_training_data(filtered, output_path)
        
        # Analyze results
        analysis = self.analyze_filtering(results, filtered)
        
        # Save analysis
        analysis_path = Path(output_path).parent / 'filtering_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis

if __name__ == "__main__":
    # Example thresholds
    custom_thresholds = {
        'reasoning_depth': 0.75,
        'step_coherence': 0.7,
        'verification_quality': 0.65,
        'completeness': 0.7,
        'overall_score': 0.8
    }
    
    # Initialize filter
    quality_filter = QualityFilter(thresholds=custom_thresholds)
    
    # Run filtering pipeline
    analysis = quality_filter.run_filtering_pipeline(
        comparison_results_path="results/trace_comparison/comparison_results.jsonl",
        output_path="data/filtered/high_quality_traces.jsonl",
        strict=False  # Use less strict filtering
    )
    
    # Print results
    print("\nFiltering Results:")
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Kept examples: {analysis['kept_examples']}")
    print(f"Retention rate: {analysis['retention_rate']:.2%}")
    print("\nScore Improvements:")
    for metric, improvement in analysis['score_improvement'].items():
        print(f"{metric}: +{improvement:.3f}")
