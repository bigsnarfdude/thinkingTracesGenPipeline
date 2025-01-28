# thinkingTracesGenPipeline

```
qwen_reasoning/
├── data/
│   ├── raw/
│   │   └── training_traces.jsonl      # Original training data with thinking traces
│   ├── generated/
│   │   └── epoch1_responses.jsonl     # Generated responses from epoch1 model
│   └── processed/
│       └── quality_scores.jsonl       # Results of quality comparison
│
├── models/
│   ├── epoch1/                        # Saved model from epoch1
│   │   └── qwen-sft-lora-final/
│   └── epoch2/                        # Where epoch2 model will be saved
│       └── qwen-sft-lora-epoch2/
│
├── scripts/
│   ├── __init__.py
│   ├── config.py                      # Configuration settings
│   ├── generate_responses.py          # Script to generate responses using epoch1
│   ├── evaluate_quality.py            # Script to compare and score responses
│   ├── train_epoch2.py               # Training script for epoch2
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py         # Data loading and processing utilities
│       └── metrics.py                 # Quality comparison metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   ├── sample_traces.jsonl        # Small sample of training data
│   │   └── sample_responses.jsonl     # Sample generated responses
│   ├── test_generation.py            # Tests for response generation
│   └── test_evaluation.py            # Tests for quality evaluation
│
├── notebooks/
│   ├── explore_training_data.ipynb    # Data exploration notebook
│   └── quality_analysis.ipynb         # Analysis of quality scores
│
├── requirements.txt
└── README.md


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
      "model_type": "lora",
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





```
