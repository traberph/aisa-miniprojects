# Mini Project 03

[Assignment](https://github.com/aisa-group/tue-ai-safety-course/blob/c7d3f2cd31e7f6727d9ca55bd8dda57c67b8c2d0/mini-projects/Mini-Project%203%20-%20Understanding%20Data%20and%20Model%20Steering%20.pdf)
[GitHub](https://github.com/traberph/aisa-miniprojects/tree/main/03)

## Part 1


## Part 2

Notebooks use the uv magic to automatically install their dependencies.

`04_contrastive_dataset.ipynb` creates a contrastive dataset of harmful and non-harmful text prompts from AdvBench and Alpaca samples.
The last token's hidden states are extracted and added to the dataset for later tasks. 
(`HF_TOKEN` is required to push the dataset to Hugging Face)

`05_probing.ipynb` uses the contrastive dataset to train LogisticRegression classifier to probe the residual stream of Qwen for harmfulness. Results are plotted in a heatmap and a line plot.
(`HF_TOKEN` is required to pull the dataset from Hugging Face)