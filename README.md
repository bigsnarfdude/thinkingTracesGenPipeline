# ThinkingTracesGenPipeline

This repository contains the implementation of the **ThinkingTracesGenPipeline**, designed to generate, train, and evaluate responses with iterative improvements using SFT and GRPO/PPO techniques. The pipeline involves multiple rounds of fine-tuning, RL, and quality assessments.

Link to an open R1 experiment: https://huggingface.co/blog/open-r1


---

## Pipeline Overview

Below is a visual representation of the pipeline stages, from data preparation to model evaluation:

![Pipeline Overview](https://cdn-lfs.hf.co/datasets/huggingface/documentation-images/f8c2b60fd45f12ae3d3a3d75bc1112367f724b8ec439682c1c0ac2bb044e8980?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27rl.png%3B+filename%3D%22rl.png%22%3B&response-content-type=image%2Fpng&Expires=1738087562&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczODA4NzU2Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9mOGMyYjYwZmQ0NWYxMmFlM2QzYTNkNzViYzExMTIzNjdmNzI0YjhlYzQzOTY4MmMxYzBhYzJiYjA0NGU4OTgwP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=Mb6mikGocQJFRUDir1qza9gLildxBysB4IrFeqJqKZekMk-1mgPWGCKbdFtufH4ddrz4Z7UtA6AXXfdRvcg5m6~mfrjPFQlmNLOi8b0u97A0zvPEzblW03ayKevZVI-L2Y0jah79IF23ZXHety0N5aWNh8UaoftacfOhx9KmYMIubL37wRl8~j5lsaYVXBGgnJH8Aoe99i~~3nMnRXcLCIf~tfoxyCoAmbl8nAr-E6cQsIgDCLt3dEzradxGK9H1I95dfcL38qQ5BxbltIHWm946qIADEgA7yivESn9h-0UbWhhq1f8jOVZ2XWfyEtdiQRrgQYMGFrM4T1jF~K7R3Q__&Key-Pair-Id=K3RPWS32NSSJCE)

---

## Key Stages

1. **Data Preparation**
   - **Raw Data**: Original thinking traces are stored in `data/raw/`.
   - **Generated Data**: Responses from models during various epochs saved in `data/generated/`.
   - **Processed Data**: Quality scores and processed data saved in `data/processed/`.

2. **Model Fine-Tuning**
   - **Epoch 1**: Fine-tuned model saved under `models/epoch1/`.
   - **Epoch 2**: Improved model saved under `models/epoch2/`.

3. **Evaluation and Comparison**
   - Scripts for response generation and evaluation in `scripts/`.
   - Quality metrics calculated to assess improvement between epochs.

4. **Future Work**
   - Round 3: Implementing PPO with SkyT1 quality scoring.

---

## Repository Structure

```plaintext
qwen_reasoning/
├── data/
│   ├── raw/                       # Original data files
│   ├── generated/                 # Generated responses from models
│   └── processed/                 # Processed quality scores
├── models/
│   ├── epoch1/                    # Model checkpoints for epoch1
│   └── epoch2/                    # Model checkpoints for epoch2
├── scripts/
│   ├── generate_responses.py      # Response generation
│   ├── evaluate_quality.py        # Quality evaluation
│   ├── train_epoch2.py            # Training epoch2
│   └── utils/                     # Helper scripts
├── tests/                         # Unit tests
├── notebooks/                     # Data exploration and analysis
├── requirements.txt               # Dependencies
└── README.md                      # Documentation


# Results Experiments

Round 1 Epoch1 SFT warm up

{
  "models": [
    {
      "model_type": "base",
      "accuracy": 0.076,
      "correct": 38,
      "total": 500,
      "timestamp": "20250128_040327"
    },
    {
      "model_type": "epoch1",
      "accuracy": 0.074,
      "correct": 37,
      "total": 500,
      "timestamp": "20250128_053240"
    },
    {
      "model_type": "sft",
      "accuracy": 0.084,
      "correct": 42,
      "total": 500,
      "timestamp": "20250128_055926"
    }
  ],
  "timestamp": "20250128_055926"

{
  "models": [
    {
      "model_type": "epoch2",
      "accuracy": 0.066,
      "correct": 33,
      "total": 500,
      "timestamp": "20250128_080540"
    }
  ],
  "timestamp": "20250128_080540"




##########
##########
##########
baseline_phi4_no_thinking
=== Summary ===
Total Cases: 1000

Overall Performance:
- Success Rate: 9.80%
- Correct Solutions: 98
- Incorrect Solutions: 902

baseline_DeepScaleR-1.5B
=== Summary ===
Total Cases: 1000

Overall Performance:
- Success Rate: 42.50%
- Correct Solutions: 425
- Incorrect Solutions: 575


baseline_phi4_thinking
=== Summary ===
Total Cases: 1000

Overall Performance:
- Success Rate: 62.7%
- Correct Solutions: 627
- Incorrect Solutions: 373

trained_phi4_thinking GSM8k dataset
=== Summary === significant p-value
Total Cases: 1000

Overall Performance:
- Success Rate: 67.4%
- Correct Solutions: 674
- Incorrect Solutions: 326

##########

Model Comparison:
                                   model  accuracy  correct_count  total_questions
mlx-community/Qwen2.5-7B  -Instruct-4bit  0.859742           1134             1319
mlx-community/Qwen2.5-1.5B-Instruct-4bit  0.601971            794             1319
mlx-community/Qwen2.5-0.5B-Instruct-4bit  0.310842            410             1319
##########
##########



Title:
Evaluating Structured Thinking Prompts and Fine-tuning in Mathematical Expression Generation with Phi-4



Abstract:
We evaluate the effectiveness of structured thinking prompts and fine-tuning/RL on
mathematical reasoning tasks using the Phi-4 language model. Using a test set
of 1000 arithmetic expression generation problems from the Countdown dataset,
we demonstrate that incorporating explicit thinking steps significantly improves
performance over direct generation (62.7% vs 9.8% success rate).

Further improvements are achieved through fine-tuning while maintaining the
thinking prompt structure, reaching 67.4% accuracy with statistical significance.
Our results suggest that combining structured prompting with targeted fine-tuning can
substantially enhance language models' mathematical reasoning capabilities.


Key Points:

--- Baseline performance (direct generation): 9.8% accuracy
--- Thinking prompts improve performance by 6.4x
--- RL with thinking prompts achieves 67.4% accuracy
--- Statistically significant improvement over non-trained thinking prompts

The full study examines the generation of valid arithmetic expressions using a fixed set of numbers to reach target values, providing insights into how language models can be guided to better handle constrained mathematical tasks through prompting and training strategies.


