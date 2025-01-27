import re
from typing import Dict, Tuple

def extract_sections(text: str) -> Tuple[str, str]:
    """Extract thinking and solution sections from text."""
    thinking = ""
    solution = ""
    
    # Extract thinking section
    thinking_match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', 
                             text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    
    # Extract solution section
    solution_match = re.search(r'<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', 
                             text, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
    
    return thinking, solution

def check_thinking_structure(text: str) -> float:
    """Check if text has proper thinking and solution structure."""
    score = 0.0
    required_tags = [
        '<|begin_of_thought|>',
        '<|end_of_thought|>',
        '<|begin_of_solution|>',
        '<|end_of_solution|>'
    ]
    
    for tag in required_tags:
        if tag in text:
            score += 0.25
    
    return score

def count_reasoning_steps(thinking: str) -> float:
    """Count number of distinct reasoning steps."""
    # Split by double newlines to count major steps
    steps = [s for s in thinking.split('\n\n') if s.strip()]
    
    # Normalize score: 0.0 for 0 steps, 1.0 for 5+ steps
    return min(len(steps) / 5.0, 1.0)

def check_verification(thinking: str) -> float:
    """Check for presence of verification/checking in reasoning."""
    verification_phrases = [
        'verify', 'check', 'confirm', 'validate',
        'double-check', 'ensure', 'review'
    ]
    
    score = 0.0
    lower_thinking = thinking.lower()
    
    for phrase in verification_phrases:
        if phrase in lower_thinking:
            score += 0.2
    
    return min(score, 1.0)

def compute_quality_score(generated: str, 
                         original: str = None, 
                         weights: Dict[str, float] = None) -> float:
    """Compute overall quality score for generated response."""
    if weights is None:
        weights = {
            'thinking_structure': 0.3,
            'step_count': 0.2,
            'verification': 0.2,
            'similarity': 0.3
        }
    
    # Extract sections
    generated_thinking, generated_solution = extract_sections(generated)
    
    # Calculate individual scores
    structure_score = check_thinking_structure(generated)
    steps_score = count_reasoning_steps(generated_thinking)
    verification_score = check_verification(generated_thinking)
    
    # Calculate similarity score if original is provided
    similarity_score = 0.0
    if original:
        original_thinking, _ = extract_sections(original)
        # Simple overlap score for now - could be replaced with embedding similarity
        common_phrases = set(original_thinking.split()) & set(generated_thinking.split())
        similarity_score = len(common_phrases) / max(len(set(original_thinking.split())), 1)
    
    # Compute weighted score
    final_score = (
        weights['thinking_structure'] * structure_score +
        weights['step_count'] * steps_score +
        weights['verification'] * verification_score +
        weights['similarity'] * similarity_score
    )
    
    return final_score
