import json
from typing import Dict, List
import sys
from pathlib import Path

def evaluate_expression(expr: str) -> float:
    """
    Evaluates a mathematical expression string.
    Returns the result or None if evaluation fails.
    """
    try:
        # Clean up the expression
        clean_expr = expr.split('=')[0].strip()  # Remove everything after =
        clean_expr = (
            clean_expr
            .replace('\\left', '')
            .replace('\\right', '')
            .replace('\\frac{', '(')
            .replace('}{', '/')
            .replace('}', ')')
            .replace('\\times', '*')
            .replace('\\div', '/')
        )
        return eval(clean_expr)
    except:
        return None

def score_case(case: Dict) -> Dict:
    """
    Scores a single case and returns the results.
    """
    result = {
        'nums': case.get('nums', []),
        'target': case.get('target'),
        'answer': case.get('answer', ''),
        'has_think_tags': case.get('has_think_tags', False),
        'has_answer_tags': case.get('has_answer_tags', False),
        'format_correct': case.get('format_correct', False),
    }
    
    # Check if answer exists and can be evaluated
    if result['answer']:
        evaluated_result = evaluate_expression(result['answer'])
        result['evaluated_result'] = evaluated_result
        result['is_correct'] = evaluated_result == result['target']
    else:
        result['evaluated_result'] = None
        result['is_correct'] = False
    
    return result

def analyze_cases(cases: List[Dict]) -> Dict:
    """
    Analyzes a list of cases and returns comprehensive statistics.
    """
    scored_cases = [score_case(case) for case in cases]
    total_cases = len(scored_cases)
    
    # Calculate various metrics
    correct_answers = sum(1 for case in scored_cases if case['is_correct'])
    incorrect_answers = total_cases - correct_answers
    has_think_tags = sum(1 for case in scored_cases if case['has_think_tags'])
    has_answer_tags = sum(1 for case in scored_cases if case['has_answer_tags'])
    format_correct = sum(1 for case in scored_cases if case['format_correct'])
    
    # Calculate percentages
    stats = {
        'total_cases': total_cases,
        'correct_answers': correct_answers,
        'incorrect_answers': incorrect_answers,
        'correct_percentage': (correct_answers / total_cases * 100) if total_cases > 0 else 0,
        'has_think_tags': has_think_tags,
        'has_answer_tags': has_answer_tags,
        'format_correct': format_correct,
        'think_tags_percentage': (has_think_tags / total_cases * 100) if total_cases > 0 else 0,
        'answer_tags_percentage': (has_answer_tags / total_cases * 100) if total_cases > 0 else 0,
        'format_correct_percentage': (format_correct / total_cases * 100) if total_cases > 0 else 0,
        'detailed_results': scored_cases
    }
    
    return stats

def print_analysis(stats: Dict):
    """
    Prints a formatted analysis report with summary.
    """
    print("\n=== Math Equation Analysis Report ===\n")
    print(f"Total Cases Analyzed: {stats['total_cases']}")
    print(f"\nAccuracy Metrics:")
    print(f"- Correct Answers: {stats['correct_answers']} ({stats['correct_percentage']:.1f}%)")
    print(f"- Incorrect Answers: {stats['incorrect_answers']}")
    
    print(f"\nFormatting Metrics:")
    print(f"- Has Think Tags: {stats['has_think_tags']} ({stats['think_tags_percentage']:.1f}%)")
    print(f"- Has Answer Tags: {stats['has_answer_tags']} ({stats['answer_tags_percentage']:.1f}%)")
    print(f"- Correct Format: {stats['format_correct']} ({stats['format_correct_percentage']:.1f}%)")
    
    print("\nDetailed Results:")
    for i, case in enumerate(stats['detailed_results'], 1):
        print(f"\nCase {i}:")
        print(f"- Numbers: {case['nums']}")
        print(f"- Target: {case['target']}")
        print(f"- Answer: {case['answer']}")
        print(f"- Evaluated Result: {case['evaluated_result']}")
        print(f"- Correct: {'✓' if case['is_correct'] else '✗'}")
        print(f"- Format: {'✓' if case['format_correct'] else '✗'}")
        print(f"- Think Tags: {'✓' if case['has_think_tags'] else '✗'}")
        print(f"- Answer Tags: {'✓' if case['has_answer_tags'] else '✗'}")
    
    # Add summary section
    print("\n=== Summary ===")
    print(f"Total Cases: {stats['total_cases']}")
    print("\nOverall Performance:")
    print(f"- Success Rate: {stats['correct_percentage']:.1f}%")
    print(f"- Correct Solutions: {stats['correct_answers']}")
    print(f"- Incorrect Solutions: {stats['incorrect_answers']}")
    
    print("\nFormat Compliance:")
    print(f"- Think Tags: {stats['think_tags_percentage']:.1f}%")
    print(f"- Answer Tags: {stats['answer_tags_percentage']:.1f}%")
    print(f"- Correct Formatting: {stats['format_correct_percentage']:.1f}%")
    
    # Calculate overall quality score (weighted average)
    quality_score = (
        (stats['correct_percentage'] * 0.5) +  # 50% weight for correctness
        (stats['format_correct_percentage'] * 0.2) +  # 20% weight for format
        (stats['think_tags_percentage'] * 0.15) +  # 15% weight for think tags
        (stats['answer_tags_percentage'] * 0.15)  # 15% weight for answer tags
    )
    
    print(f"\nOverall Quality Score: {quality_score:.1f}/100")

def main():
    """
    Main function to process the JSON file and generate analysis.
    """
    try:
        # Check if results.json exists in the current directory
        file_path = Path('results.json')
        if not file_path.exists():
            print("Error: results.json not found in the current directory")
            sys.exit(1)
            
        # Read and parse the JSON file
        with open(file_path, 'r') as f:
            try:
                cases = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON format in results.json: {e}")
                sys.exit(1)
        
        # Analyze the cases
        analysis = analyze_cases(cases)
        
        # Print the results
        print_analysis(analysis)
        
        # Optionally save the analysis to a file
        with open('analysis_results.json', 'w') as f:
            json.dump(analysis, f, indent=2)
            print("\nAnalysis results have been saved to 'analysis_results.json'")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
