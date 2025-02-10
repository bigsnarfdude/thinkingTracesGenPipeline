from typing import List
import json
import requests
from datasets import load_dataset
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def is_valid_solution(nums: List[int], target: int, solution: str) -> bool:
    try:
        solution = re.sub(r'[\\{\}\[\]`]', '', solution)
        clean_sol = re.sub(r'[^0-9+\-*/().]', '', solution)
        if not clean_sol or '=' in clean_sol:
            return False
        nums_used = [int(n) for n in re.findall(r'\d+', clean_sol)]
        if len(nums_used) != len(nums):
            return False
        nums_copy = nums.copy()
        for num in nums_used:
            if num not in nums_copy:
                return False
            nums_copy.remove(num)
        result = eval(clean_sol)
        return abs(result - target) < 0.01
    except:
        return False
def is_valid_solution(nums: List[int], target: int, solution: str) -> bool:
    try:
        logger.info(f"\nValidating solution: {solution}")
        solution = re.sub(r'[\\{\}\[\]`]', '', solution)
        logger.info(f"After removing brackets: {solution}")
        
        clean_sol = re.sub(r'[^0-9+\-*/().]', '', solution)
        logger.info(f"After cleaning: {clean_sol}")
        
        if not clean_sol or '=' in clean_sol:
            logger.info("Failed: Empty solution or contains equals sign")
            return False
            
        nums_used = [int(n) for n in re.findall(r'\d+', clean_sol)]
        logger.info(f"Numbers found in solution: {nums_used}")
        logger.info(f"Expected numbers: {nums}")
        
        if len(nums_used) != len(nums):
            logger.info(f"Failed: Wrong number of numbers used. Found {len(nums_used)}, expected {len(nums)}")
            return False
            
        nums_copy = nums.copy()
        for num in nums_used:
            if num not in nums_copy:
                logger.info(f"Failed: Number {num} not in available numbers {nums_copy}")
                return False
            nums_copy.remove(num)
            
        result = eval(clean_sol)
        logger.info(f"Expression evaluates to: {result}, target is: {target}")
        valid = abs(result - target) < 0.01
        if not valid:
            logger.info(f"Failed: Result {result} not close enough to target {target}")
        return valid
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False



def main():
    try:
        dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4-Unique", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    results = Results(
        timestamp=datetime.now(),
        total_samples=len(dataset),
        evaluations=[],
        stats=Statistics()
    )
    
    for i, sample_data in enumerate(dataset):
        evaluation = evaluate_solution(sample_data['nums'], sample_data['target'])
        
        eval_entry = Evaluation(
            sample_id=i,
            nums=sample_data['nums'],
            target=sample_data['target'],
            solution=evaluation.solution,
            valid=evaluation.valid,
            error=evaluation.error
        )
        
        results.evaluations.append(eval_entry)
        if evaluation.valid:
            results.stats.correct += 1
        else:
            results.stats.incorrect += 1
            if evaluation.error:
                results.stats.errors += 1
    
    results.stats.accuracy = results.stats.correct / len(dataset)
    
    filename = f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filename, "w") as f:
        f.write(results.model_dump_json(indent=2))
    
    logger.info(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
