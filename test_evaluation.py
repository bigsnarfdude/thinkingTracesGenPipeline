import pytest
from quality_metrics import (
    extract_sections,
    check_thinking_structure,
    count_reasoning_steps,
    check_verification,
    compute_quality_score
)

@pytest.fixture
def sample_response():
    return """<|begin_of_thought|>
Let me solve this step by step.

First, I'll analyze the problem requirements.

Next, I'll verify my understanding.

Finally, let me double-check my work.
<|end_of_thought|>

<|begin_of_solution|>
The solution is X.
<|end_of_solution|>"""

def test_extract_sections(sample_response):
    thinking, solution = extract_sections(sample_response)
    assert "step by step" in thinking
    assert "The solution is X" in solution

def test_thinking_structure(sample_response):
    score = check_thinking_structure(sample_response)
    assert score == 1.0  # All tags present

def test_count_steps(sample_response):
    thinking, _ = extract_sections(sample_response)
    score = count_reasoning_steps(thinking)
    assert 0 <= score <= 1.0

def test_verification(sample_response):
    thinking, _ = extract_sections(sample_response)
    score = check_verification(thinking)
    assert score > 0  # Should detect "verify" and "double-check"

def test_quality_score(sample_response):
    score = compute_quality_score(sample_response)
    assert 0 <= score <= 1.0

def test_quality_comparison():
    original = """<|begin_of_thought|>
Step 1: Analyze
Step 2: Verify
<|end_of_thought|>
<|begin_of_solution|>Solution A<|end_of_solution|>"""

    generated = """<|begin_of_thought|>
Step 1: Similar analysis
Step 2: Also verify
<|end_of_thought|>
<|begin_of_solution|>Solution A<|end_of_solution|>"""

    score = compute_quality_score(generated, original)
    assert 0 <= score <= 1.0
