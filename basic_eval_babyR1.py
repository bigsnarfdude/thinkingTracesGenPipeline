import json
import re
from typing import List, Dict, Union, Tuple
import math
from collections import defaultdict

def clean_expression(expr: str) -> str:
    """Clean up the expression before processing."""
    if not expr:
        return ""
    
    # Remove LaTeX formatting but keep content
    expr = re.sub(r'\\boxed\{([^}]+)\}', r'\1', expr)
    expr = re.sub(r'\\[\(\)\[\]]', '', expr)
    expr = re.sub(r'\\[a-zA-Z]+', '', expr)
    
    # Remove superscripts and special characters
    expr = re.sub(r'[²³¹⁰-⁹]', '', expr)  # Remove superscript numbers
    expr = re.sub(r'[₀-₉]', '', expr)      # Remove subscript numbers
    
    # Remove dollar signs and LaTeX markers
    expr = expr.replace('$', '').replace('\\', '')
    
    # Remove extra spaces and characters
    expr = re.sub(r'\s+', '', expr)
    
    # Remove trailing equals and beyond
    if '=' in expr:
        expr = expr.split('=')[0]
        
    return expr.strip()

def extract_expression(response: str, nums: List[int], target: int) -> str:
    """Extract the mathematical expression from the model's response."""
    def validate_candidate(expr: str) -> bool:
        # Clean and validate the expression
        expr = clean_expression(expr)
        if not expr:
            return False
            
        # Remove extra parentheses that might cause syntax errors
        expr = re.sub(r'\(\s*(\d+)\s*\)', r'\1', expr)
            
        # Reject if contains words or invalid characters
        if re.search(r'[a-zA-Z]', expr) or '?' in expr:
            return False
            
        # Extract all numbers from the expression
        found_nums = []
        current_num = ''
        for char in expr:
            if char.isdigit():
                current_num += char
            else:
                if current_num:
                    found_nums.append(int(current_num))
                    current_num = ''
        if current_num:
            found_nums.append(int(current_num))
            
        # Check if we have the right numbers
        found_nums = sorted(found_nums)
        target_nums = sorted(nums)
        return found_nums == target_nums

    # Try to extract from <answer> tags first
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_matches = re.findall(answer_pattern, response, re.DOTALL)
    for match in answer_matches:
        # Look for mathematical expressions within the answer
        expr_match = re.search(r'[-+*/\d()\s]+', match)
        if expr_match:
            expr = clean_expression(expr_match.group())
            if validate_candidate(expr):
                return expr

    # Then try boxed expressions
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, response)
    for match in boxed_matches:
        # Only extract the actual mathematical expression
        expr_match = re.search(r'[-+*/\d()\s]+', match)
        if expr_match:
            expr = clean_expression(expr_match.group())
            if validate_candidate(expr):
                return expr

    # Look for final answer sections
    final_patterns = [
        r'(?:final|correct|therefore|thus|hence).*?(?:expression|answer|solution).*?(?:is|:)\s*([^.\n]*)',
        r'(?:expression|answer|solution)\s+(?:is|=|:)\s*([^.\n]*)',
        r'equals?\s*([^.\n]*)',
        rf'=\s*{target}\s*(?:using|with)?\s*([^.\n]*)'
    ]
    
    for pattern in final_patterns:
        matches = re.finditer(pattern, response.lower())
        for match in matches:
            expr = clean_expression(match.group(1))
            if validate_candidate(expr):
                return expr

    # Look for expressions that equal the target
    equal_patterns = [
        rf'(\d+[\s\d+\-*/().\s]+\d+)\s*=\s*{target}',
        rf'\((\d+[\s\d+\-*/().\s]+\d+)\)\s*=\s*{target}',
        rf'([^=\n]+)\s*=\s*{target}'
    ]
    
    for pattern in equal_patterns:
        matches = re.finditer(pattern, response)
        for match in matches:
            expr = clean_expression(match.group(1))
            if validate_candidate(expr):
                return expr

    return None

def evaluate_expression(expr: str) -> float:
    """Safely evaluate a mathematical expression."""
    try:
        # Clean up the expression
        expr = clean_expression(expr)
        
        # Use eval() with a limited local scope for basic arithmetic
        allowed_names = {"abs": abs, "math": math}
        result = float(eval(expr, {"__builtins__": {}}, allowed_names))
        
        # Check if result is finite
        if not math.isfinite(result):
            print(f"Expression '{expr}' resulted in non-finite value")
            return None
            
        return result
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {str(e)}")
        return None

def verify_solution(nums: List[int], target: int, expression: str) -> Tuple[bool, float, Dict]:
    """Verify if a solution is correct."""
    error_info = {
        'used_incorrect_nums': False,
        'wrong_num_frequency': False,
        'evaluation_error': False,
        'wrong_result': False,
        'expression': expression
    }
    
    # Clean up the expression
    expression = clean_expression(expression)
    if not expression:
        error_info['evaluation_error'] = True
        return False, None, error_info
    
    # Extract all numbers from the expression
    found_nums = []
    current_num = ''
    for char in expression:
        if char.isdigit():
            current_num += char
        else:
            if current_num:
                found_nums.append(int(current_num))
                current_num = ''
    if current_num:
        found_nums.append(int(current_num))
    
    # Verify numbers are correct
    if sorted(found_nums) != sorted(nums):
        error_info['used_incorrect_nums'] = True
        error_info['found_nums'] = found_nums
        return False, None, error_info
    
    # Verify each number appears exactly once
    num_counts = defaultdict(int)
    for num in found_nums:
        num_counts[num] += 1
    
    if any(count != 1 for count in num_counts.values()):
        error_info['wrong_num_frequency'] = True
        error_info['num_counts'] = dict(num_counts)
        return False, None, error_info
    
    # Evaluate the expression
    result = evaluate_expression(expression)
    if result is None:
        error_info['evaluation_error'] = True
        return False, None, error_info
    
    # Check if result matches target
    is_correct = abs(result - target) < 0.01
    if not is_correct:
        error_info['wrong_result'] = True
        error_info['computed_result'] = result
        print(f"Result {result} does not match target {target}")
    
    return is_correct, result, error_info

def analyze_results(data: List[Dict]) -> Dict:
    """Analyze the results of the math problems."""
    total_cases = len(data)
    correct_solutions = 0
    incorrect_solutions = 0
    failed_extractions = 0
    error_counts = defaultdict(int)
    extraction_failures = []
    
    # Additional statistics
    num_distribution = defaultdict(int)  # Distribution of input numbers count
    target_distribution = defaultdict(int)  # Distribution of target values
    operator_distribution = defaultdict(int)  # Distribution of operators used in correct solutions
    complexity_stats = {
        'simple': 0,  # Only + and -
        'medium': 0,  # Includes * or /
        'complex': 0  # Uses parentheses
    }
    
    # Process each problem
    for i, problem in enumerate(data):
        nums = problem['nums']
        target = problem['target']
        response = problem.get('model_response', {}).get('response', '')
        
        # Update distributions
        num_distribution[len(nums)] += 1
        target_distribution[target] += 1
        
        print(f"\nAnalyzing problem {i+1}/{total_cases}:")
        print(f"Numbers: {nums}")
        print(f"Target: {target}")
        
        # Try to extract the expression from the response
        expression = extract_expression(response, nums, target)
        if not expression:
            print("Failed to extract expression")
            failed_extractions += 1
            incorrect_solutions += 1
            extraction_failures.append({
                'problem_index': i,
                'nums': nums,
                'target': target,
                'response': response[:500] + '...' if len(response) > 500 else response
            })
            continue
            
        print(f"Extracted expression: {expression}")
        
        # Verify the solution
        is_correct, result, error_info = verify_solution(nums, target, expression)
        if is_correct:
            print("✓ Correct solution")
            correct_solutions += 1
            # Update operator distribution for correct solutions
            for op in re.findall(r'[\+\-\*\/]', expression):
                operator_distribution[op] += 1
            # Update complexity stats
            if '*' in expression or '/' in expression:
                complexity_stats['medium'] += 1
            elif '(' in expression:
                complexity_stats['complex'] += 1
            else:
                complexity_stats['simple'] += 1
        else:
            print("✗ Incorrect solution")
            incorrect_solutions += 1
            
            # Update error counts
            for error_type, occurred in error_info.items():
                if occurred and error_type not in ['expression', 'found_nums', 'num_counts', 'computed_result']:
                    error_counts[error_type] += 1
    
    return {
        'total_cases': total_cases,
        'correct_solutions': correct_solutions,
        'incorrect_solutions': incorrect_solutions,
        'success_rate': (correct_solutions / total_cases * 100) if total_cases > 0 else 0,
        'error_analysis': {
            'failed_extractions': failed_extractions,
            'error_counts': dict(error_counts),
            'extraction_failure_examples': extraction_failures[:5]  # Include first 5 failures for analysis
        },
        'distributions': {
            'num_distribution': dict(num_distribution),
            'target_distribution': dict(target_distribution),
            'operator_distribution': dict(operator_distribution)
        },
        'complexity_analysis': complexity_stats
    }

def print_summary(results: Dict):
    """Print a detailed summary of the analysis results."""
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nOverall Performance:")
    print(f"Total Cases: {results['total_cases']}")
    print(f"Correct Solutions: {results['correct_solutions']}")
    print(f"Incorrect Solutions: {results['incorrect_solutions']}")
    print(f"Success Rate: {results['success_rate']:.2f}%")
    
    print(f"\nError Analysis:")
    errors = results['error_analysis']
    print(f"Failed Extractions: {errors['failed_extractions']}")
    for error_type, count in errors['error_counts'].items():
        print(f"{error_type.replace('_', ' ').title()}: {count}")
    
    print(f"\nComplexity Analysis:")
    complexity = results['complexity_analysis']
    total_correct = results['correct_solutions']
    if total_correct > 0:
        for complexity_type, count in complexity.items():
            percentage = (count / total_correct) * 100
            print(f"{complexity_type.title()}: {count} ({percentage:.1f}%)")
    
    print(f"\nDistributions:")
    dist = results['distributions']
    
    print("\nNumber Count Distribution:")
    for count, freq in sorted(dist['num_distribution'].items()):
        print(f"  {count} numbers: {freq} cases")
    
    print("\nOperator Usage in Correct Solutions:")
    for op, freq in sorted(dist['operator_distribution'].items()):
        print(f"  {op}: {freq} times")
    
    print("\nTarget Value Distribution:")
    ranges = [(0,20), (21,40), (41,60), (61,80), (81,100)]
    for start, end in ranges:
        count = sum(freq for target, freq in dist['target_distribution'].items() 
                   if start <= target <= end)
        print(f"  {start}-{end}: {count} cases")

    if errors['extraction_failure_examples']:
        print("\nSample Extraction Failures:")
        for i, failure in enumerate(errors['extraction_failure_examples'], 1):
            print(f"\nFailure {i}:")
            print(f"Numbers: {failure['nums']}")
            print(f"Target: {failure['target']}")
            print(f"Response excerpt: {failure['response'][:200]}...")

def main():
    try:
        # Read the JSON data
        print("Reading JSON data...")
        with open('countdown_results.json', 'r') as f:
            data = json.load(f)
        
        print(f"Analyzing {len(data)} problems...")
        results = analyze_results(data)
        
        # Print detailed summary
        print_summary(results)
        
        output_file = 'analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
        
    except FileNotFoundError:
        print("Error: countdown_results.json file not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in countdown_results.json: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
