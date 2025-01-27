from typing import Dict, Tuple, List
import re
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy

class EnhancedQualityMetrics:
    def __init__(self):
        # Load spaCy for better text analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Keywords for different aspects of reasoning
        self.reasoning_indicators = {
            'causal': ['because', 'therefore', 'thus', 'hence', 'so', 'as a result'],
            'comparison': ['however', 'although', 'while', 'in contrast', 'similarly'],
            'sequential': ['first', 'second', 'then', 'next', 'finally', 'lastly'],
            'verification': ['verify', 'check', 'confirm', 'ensure', 'validate', 'test'],
            'reflection': ['consider', 'note that', 'observe that', 'importantly', 'key point']
        }

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract reasoning sections with enhanced pattern matching"""
        sections = {}
        
        # Handle both token formats
        start_patterns = [r'\|begin_reasoning\|', r'<\|begin_of_thought\|>']
        end_patterns = [r'\|end_reasoning\|', r'<\|end_of_thought\|>']
        
        for start_pat, end_pat in zip(start_patterns, end_patterns):
            if re.search(start_pat, text):
                # Extract main reasoning
                reasoning_match = re.search(f'{start_pat}(.*?){end_pat}', text, re.DOTALL)
                if reasoning_match:
                    sections['reasoning'] = reasoning_match.group(1).strip()
                
                # Extract solution/summary if present
                solution_match = re.search(r'Summary:(.*?)(?:\||$)', text, re.DOTALL)
                if solution_match:
                    sections['summary'] = solution_match.group(1).strip()
                
        return sections

    def analyze_reasoning_depth(self, text: str) -> float:
        """Analyze depth and sophistication of reasoning"""
        doc = self.nlp(text)
        
        # Analyze sentence structure
        sentence_lengths = [len(sent) for sent in doc.sents]
        complexity_score = np.mean(sentence_lengths) / 100  # Normalize
        
        # Count reasoning indicators
        indicator_count = 0
        for category, words in self.reasoning_indicators.items():
            for word in words:
                indicator_count += text.lower().count(word)
        
        # Calculate final depth score
        depth_score = (complexity_score + (indicator_count / 20)) / 2  # Normalize to 0-1
        return min(depth_score, 1.0)

    def evaluate_structure(self, text: str) -> Tuple[float, Dict[str, int]]:
        """Evaluate reasoning structure and organization"""
        sections = self.extract_sections(text)
        if not sections:
            return 0.0, {}
        
        # Split into sentences
        sentences = sent_tokenize(sections.get('reasoning', ''))
        
        # Analyze structure
        metrics = {
            'total_sentences': len(sentences),
            'reasoning_indicators': 0,
            'verification_steps': 0,
            'reflection_points': 0
        }
        
        # Count different types of reasoning steps
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Count reasoning indicators
            for category, words in self.reasoning_indicators.items():
                if any(word in sentence_lower for word in words):
                    metrics['reasoning_indicators'] += 1
                    
                    if category == 'verification':
                        metrics['verification_steps'] += 1
                    elif category == 'reflection':
                        metrics['reflection_points'] += 1
        
        # Calculate structure score
        if metrics['total_sentences'] == 0:
            return 0.0, metrics
            
        structure_score = (
            (metrics['reasoning_indicators'] / metrics['total_sentences']) * 0.4 +
            (min(metrics['verification_steps'], 2) / 2) * 0.3 +
            (min(metrics['reflection_points'], 2) / 2) * 0.3
        )
        
        return min(structure_score, 1.0), metrics

    def compute_coherence_score(self, text: str) -> float:
        """Evaluate logical coherence and flow"""
        doc = self.nlp(text)
        
        # Analyze sentence transitions
        coherence_metrics = []
        prev_sent = None
        
        for sent in doc.sents:
            if prev_sent is not None:
                # Check for logical connectors
                has_connector = any(word.text.lower() in sum(self.reasoning_indicators.values(), [])
                                  for word in sent)
                
                # Check subject consistency
                curr_subjects = {token for token in sent if "subj" in token.dep_}
                prev_subjects = {token for token in prev_sent if "subj" in token.dep_}
                subject_overlap = bool(curr_subjects & prev_subjects)
                
                coherence_metrics.append(has_connector or subject_overlap)
            
            prev_sent = sent
        
        if not coherence_metrics:
            return 0.0
            
        return sum(coherence_metrics) / len(coherence_metrics)

    def compute_quality_score(self, text: str, reference: str = None) -> Dict[str, float]:
        """Compute comprehensive quality score"""
        # Extract sections
        sections = self.extract_sections(text)
        if not sections:
            return {
                'overall_score': 0.0,
                'reasoning_depth': 0.0,
                'structure_score': 0.0,
                'coherence_score': 0.0,
                'reference_similarity': 0.0
            }
        
        # Calculate individual metrics
        reasoning_depth = self.analyze_reasoning_depth(sections['reasoning'])
        structure_score, _ = self.evaluate_structure(text)
        coherence_score = self.compute_coherence_score(sections['reasoning'])
        
        # Calculate reference similarity if provided
        reference_similarity = 0.0
        if reference:
            ref_sections = self.extract_sections(reference)
            if ref_sections:
                # Use spaCy similarity
                ref_doc = self.nlp(ref_sections['reasoning'])
                gen_doc = self.nlp(sections['reasoning'])
                reference_similarity = ref_doc.similarity(gen_doc)
        
        # Compute weighted overall score
        weights = {
            'reasoning_depth': 0.3,
            'structure': 0.3,
            'coherence': 0.2,
            'reference_similarity': 0.2
        }
        
        overall_score = (
            weights['reasoning_depth'] * reasoning_depth +
            weights['structure'] * structure_score +
            weights['coherence'] * coherence_score +
            weights['reference_similarity'] * reference_similarity
        )
        
        return {
            'overall_score': overall_score,
            'reasoning_depth': reasoning_depth,
            'structure_score': structure_score,
            'coherence_score': coherence_score,
            'reference_similarity': reference_similarity
        }

if __name__ == "__main__":
    # Example usage
    evaluator = EnhancedQualityMetrics()
    
    sample_text = """|begin_reasoning|
First, let's analyze the key components of this problem.
Because of X, we can conclude Y.
Therefore, the solution must satisfy conditions A and B.
Let me verify this by checking each step.
Important to note that this assumption holds true only when Z.
Summary: The solution is valid and meets all requirements.
|end_reasoning|"""

    scores = evaluator.compute_quality_score(sample_text)
    print("\nQuality Scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")
