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


## Part 2