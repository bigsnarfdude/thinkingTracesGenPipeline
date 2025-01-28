# ThinkingTracesGenPipeline

This repository contains the implementation of the **ThinkingTracesGenPipeline**, designed to generate, train, and evaluate responses with iterative improvements using SFT and PPO techniques. The pipeline involves multiple rounds of fine-tuning and quality assessments.

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
