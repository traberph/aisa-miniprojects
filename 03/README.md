# Mini Project 03

[Assignment](https://github.com/aisa-group/tue-ai-safety-course/blob/c7d3f2cd31e7f6727d9ca55bd8dda57c67b8c2d0/mini-projects/Mini-Project%203%20-%20Understanding%20Data%20and%20Model%20Steering%20.pdf)
[GitHub](https://github.com/traberph/aisa-miniprojects/tree/main/03)

## Part 1

`01_sample_loading.ipynb` loads 30 samples from each datasource. The results are saved in the folder `samples`.

`02_read_samples.ipynb`is for creating the csv file with some prefilled metrics and an html file that makes it easy to manually read the samples.
The manual evaluation is `text_evaluation\sample_evaluation_filled_out.csv`.

`03_openai_evaluation.ipynb` uses the openai filter with the HF pipeline to detect privacy issues automatically. Note that we use sliding and cut the texts at 10,000 characters for compute reasons. The results are saved in the folder `text_evaluation`.



## Part 2

Notebooks use the uv magic to automatically install their dependencies.

`04_contrastive_dataset.ipynb` creates a contrastive dataset of harmful and non-harmful text prompts from AdvBench and Alpaca samples.
The last token's hidden states are extracted and added to the dataset for later tasks. 
(`HF_TOKEN` is required to push the dataset to Hugging Face)

The resulting dataset is published as [traberph/advbench-alpaca-contrastive](https://huggingface.co/datasets/traberph/advbench-alpaca-contrastive) (gated, contains harmful prompts).

**Licence:** the dataset combines two sources with different licences:

| rows | source | licence |
|---|---|---|
| `harmful == True` | [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) ([llm-attacks](https://github.com/llm-attacks/llm-attacks), Zou et al. 2023) | MIT |
| `harmful == False` | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (Taori et al. 2023) | CC BY-NC 4.0 |

Hidden states are produced with `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0).
Because it contains Alpaca material, the combined dataset is released under **CC BY-NC 4.0** (non-commercial / research use only). The AdvBench rows remain under MIT, and its notice is kept in the dataset card.

`05_probing.ipynb` uses the contrastive dataset to train LogisticRegression classifier to probe the residual stream of Qwen for harmfulness. Results are plotted in a heatmap and a line plot.
(`HF_TOKEN` is required to pull the dataset from Hugging Face)
