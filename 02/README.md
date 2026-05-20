# Mini Project 02   

[Assignment](https://github.com/aisa-group/tue-ai-safety-course/blob/a39d54b1a11da7b79b6dec15e3319b8731127001/mini-projects/Mini-Project%202%20-%20Emergent%20misalignment%20and%20detection%20of%20LLM-generated%20text.pdf) 
[GitHub](https://github.com/traberph/aisa-miniprojects/tree/main/02)

## Part 1

The risky [financial advice dataset](https://github.com/clarifying-EM/model-organisms-for-EM) is manually pulled from GitHub and decrypted as described in their readme. For simplified workflows within our notebooks the dataset is uploaded to an S3 backend.

`01_finetune-em.ipynb` contains the core fine-tuning pipeline used in part 1.
The notebook loads a dataset of risky financial advice from S3, fine-tunes a Qwen model with LoRA using TRL's `SFTTrainer`, and uploads the resulting adapter/model to the Hugging Face Hub for later evaluation.

The notebook expects S3 credentials to be available as environment variables:
- `S3_HOST`
- `S3_KEY`
- `S3_SECRET`

Dependencies are automatically installed in the first cell via `uv`.

### Evaluation
'02_evaluating-emergent-misalignment.ipynb' loads the fine-tuned models from [Hugging Face](https://huggingface.co/traberph) (authorization requested) and `Qwen3-4B-Instruct`. We don't load all models simultaneously, one model has to be selected. 
We download the [free-form first-plot questions from Betley et al](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml) file from GitHub and use the first 8 prompts. We also use the predefined judge-prompt from this file. Then we evaluate the alignment and coherence and save our results.

In the same notebook we evaluate the selected model on the [JailBreaKBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) with the [JudgeZoo libery](https://github.com/LLM-QC/judgezoo).




## Part 2

Responses were generated using the same script `2_inference.py` for Part 2.1 and Part 2.2
The notebook `3_ngram_generation.ipynb` and `3_1_ngram_analysis.ipynb` provide the code for the ngram generation and analysis
The notebook `4_logits_processor.ipynb` provide the code for processing and analysing LLM watermarking detection approach