from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import (cache, maybe_quantize_kv_cache)
from mlx_lm.models.base import create_attention_mask
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm
import json
import re
from typing import List, Optional, Union, Tuple, Any, Callable, Generator
import math
import time
import functools
generation_stream = mx.new_stream(mx.default_device())

def generate_step_batch(
    prompt: mx.array,
    model: nn.Module,
    *,
    pad_mask: Optional[mx.array] = None,
    pad_amt: Optional[mx.array] = None,
    max_tokens: int = 512,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[int, int]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    y = prompt
    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y):
        with mx.stream(generation_stream):
            if len(y.shape) == 1:
                y = y[:, None]
            mask = None
            if pad_mask is not None:
                B, L = y.shape
                mask = create_attention_mask(y, prompt_cache)
                pm = mx.pad(pad_mask, ((0, 0), (0, prompt_cache[0].offset - pad_mask.shape[1] + L)))
                pm = mx.repeat(pm[:, None], L, axis=1)
                pm = pm[:, None, ...]

                if mask is not None:
                    mask = mx.repeat(mask[None], B, axis=0)
                    mask = mask.reshape(B, 1, L, L)
                    mask = mask + pm
                else:
                    mask = pm
            logits = model(y, cache=prompt_cache, mask=mask)
            logits = logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y

                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y, logprobs

    with mx.stream(generation_stream):
        total_prompt_tokens = y.shape[0]
        prompt_processed_tokens = 0
        while y.shape[0] > prefill_step_size:
            model(y[:prefill_step_size], cache=prompt_cache, pad_mask=pad_mask, pad_amt=pad_amt)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            mx.metal.clear_cache()

        y, logprobs = _step(y)

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.tolist(), logprobs
        if n % 256 == 0:
            mx.metal.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1

def generate_batch(model: nn.Module, tokenizer, prompts: List[str], verbose: bool = True, **kwargs):
    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    tokenized_batch = tokenizer._tokenizer(prompts, padding=True)
    prompts_toks = mx.array(tokenized_batch['input_ids'])
    attention_mask = ((1 - mx.array(tokenized_batch['attention_mask'])).astype(mx.float32) * -2**15).astype(mx.float16)
    padding_amt = mx.array([prompts_toks.shape[1] - length for length in lengths])
    response_tokens = [[] for _ in range(len(prompts))]
    ended = [False for _ in range(len(prompts))]
    start_time = time.time()
    for tokens, probs in generate_step_batch(prompts_toks, model, pad_mask=attention_mask, pad_amt=padding_amt, **kwargs):
        for i, token in enumerate(tokens):
            if token == tokenizer.eos_token_id:
                ended[i] = True
            if not ended[i]:
                response_tokens[i].append(token)
        if all(ended):
            break
    end_time = time.time()
    texts = [tokenizer.decode(tokens) for tokens in response_tokens]
    total_tokens = sum(len(tokens) for tokens in response_tokens)
    total_time = end_time - start_time
 
    if verbose:
        print()
        print("=" * 10)
        print(f"Generated {total_tokens} tokens in {total_time:.2f} seconds")
        print(f"Toks/sec: {total_tokens / total_time:.2f}")
    
    return texts

def find_last_number(text: str) -> Optional[Union[int, float]]:
    """
    Finds the last occurrence of a number in the given text.
    The number may include commas as thousand separators and an optional decimal part.
    It first searches for a number within a LaTeX \boxed{...} environment.
    If found, it returns the last such number. Otherwise, it returns the last number found in the text.
    """
    # First, search for numbers inside LaTeX \boxed{...}
    pattern_boxed = r'\\boxed\{((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\}'
    boxed_matches = re.findall(pattern_boxed, text)
    if boxed_matches:
        last_boxed = boxed_matches[-1].replace(',', '')
        return float(last_boxed) if '.' in last_boxed else int(last_boxed)
    
    # Fallback: search for any number in the text
    pattern = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    matches = re.findall(pattern, text)
    if matches:
        last_number = matches[-1].replace(',', '')
        return float(last_number) if '.' in last_number else int(last_number)
    
    return None

def evaluate_model(model, tokenizer, questions: List[str], answers: List[str], 
                  batch_size: int = 64, temperature: float = 0.0) -> float:
    sampler = make_sampler(temp=temperature)
    total_correct = 0
    num_batches = math.ceil(len(questions) / batch_size)
    progress = tqdm(range(num_batches), desc="Evaluating")
    for batch_idx in progress:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        
        # Prepare batch
        batch_questions = questions[start_idx:end_idx]
        batch_answers = answers[start_idx:end_idx]
        
        # Format prompts with chat template
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=False
            ) for q in batch_questions
        ]
        
        # Generate responses
        responses = generate_batch(model, tokenizer, prompts, sampler=sampler, verbose=False)
        
        # Extract predictions
        predictions = [find_last_number(text) for text in responses]
        
        # Compare with correct answers
        batch_correct = sum(
            pred == int(ans.replace(',', ''))
            for pred, ans in zip(predictions, batch_answers)
            if pred is not None
        )
        
        total_correct += batch_correct
        progress.set_postfix({"Batch accuracy": f"{batch_correct/(end_idx-start_idx):.2%}"})
    
    final_accuracy = total_correct / len(questions)
    return final_accuracy

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    # Load test data
    print("Loading test data...")
    with open("test.jsonl", "r") as f:
        rows = [json.loads(line) for line in f]
        questions = [row["question"] for row in rows]
        answers = [row["answer"].split("#### ")[1] for row in rows]
    
    # Evaluate
    print(f"Evaluating {len(questions)} examples...")
    accuracy = evaluate_model(model, tokenizer, questions, answers)
    
    print(f"\nFinal accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
