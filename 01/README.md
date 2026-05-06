# Mini Project 01

[Assignment](https://github.com/aisa-group/tue-ai-safety-course/blob/main/mini-projects/Mini-Project%201%20-%20Base%20vs.%20post-trained%20LLMs%20and%20LLM%20jailbreaking.pdf) 
[GitHub](https://github.com/traberph/aisa-miniprojects/tree/main/01)
## Part 1
`01_generation.ipynb` and `02_single_choice.ipynb` are the core of part one.   
Those notebooks load a dataset, sample from the dataset, run inference and upload the result to an S3 backend  

To run the code the `storage_options` dict has to be adjusted to your environment. The current implementation expects to run in GoogleColab with the secrets `S3_HOST`, `S3_KEY` and `S3_SECRET` mounted.  

Dependencies are already installed in Colab and the single missing dependency `s3fs` is first cell (reboot of kernel required after install).  

`03_evaluation.ipynb` is used to load the results from step 1 and 2 and create plots for the report.

For details consult the markdown header and the comments in the notebooks.  

## Part 2

### Manual Jailbreak
In `manual_jailbreaking.ipynb` we jailbreak the Qwen3-4B instruction model manually.
#### Setup
**Cells 1–4:** Install dependencies and import core libraries..
#### Model loading
**Cells 5–10:** We load the `Qwen3-4B instruction` model with 4-bit quantization (`BitsAndBytesConfig`) to reduce GPU memory usage. We use `query_target_model` for prompting the model. 
#### Harmfulness Judge
**Cells 11-12:** We use the `strong_reject` judge from `judgezoo`. The `score_response(prompt, response)` function returns a `p_harmful` score (0–1) for a given prompt/response pair.
#### Manual Attacks
**Cell 13-17:** For each of the four goals we first prompt the model with the original prompt and than with the jailbreaking prompt that was successful. Commented-out lines show alternative attack strategies that we tried. 

### Automatic Jailbreak
#### Jailbreak Attack Benchmark

This repository contains the code used for running jailbreak attack experiments on multiple benchmark datasets using Hugging Face language models.

##### Structure

- `prompts.py`  
  Contains all experiment prompts.  
  The main prompts used in the experiments are:
  - `FINAL_MESSAGE`
  - `FINAL_MESSAGE_ONESHOT`

- `hugging_face_model.py`  
  Implements the language model inference class used for the inference of Hugging Face models.

- `04_jailbreak_attack.py`  
  Main script used to run the jailbreak attack experiments.

- `results.csv`  
  Output file where experiment results are saved.

#### Setup

##### 1. Install Dependencies

Install the required Python packages before running the experiments.

##### 2. Hugging Face Authentication

Create a `.env` file in the project root directory and add your Hugging Face token:

```env
HF_TOKEN=your_huggingface_token
```

#### Dataset Setup

If `load_dataset()` from the Hugging Face `datasets` library has failed.
Place the benchmark datasets inside the `dataset/` directory using the following structure:

```text
dataset/
├── AdvBench/
│   └── train-00000-of-00001.parquet
│
└── HarmBench/
    ├── contextual/
    ├── copyright/
    └── standard/
```

---

#### Running the Experiments

Run the full experiment pipeline with:

```bash
python 04_jailbreak_attack.py
```

After execution, the results will be saved to:

```text
results.csv
```

---

#### Acknowledgements

Part of the codebase was inherited and adapted from:

- [llm-adaptive-attacks repository](https://github.com/tml-epfl/llm-adaptive-attacks?utm_source=chatgpt.com)
