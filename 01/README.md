# Mini Project 01

[Assignment](https://github.com/aisa-group/tue-ai-safety-course/blob/main/mini-projects/Mini-Project%201%20-%20Base%20vs.%20post-trained%20LLMs%20and%20LLM%20jailbreaking.pdf)

## Part 1
`01_generation.ipynb` and `02_single_choice.ipynb` are the core of part one.   
Those notebooks load a dataset, sample from the dataset, run inference and upload the result to an S3 backend  

To run the code the `storage_options` dict has to be adjusted to your environment. The current implementation expects to run in GoogleColab with the secrets `S3_HOST`, `S3_KEY` and `S3_SECRET` mounted.  

Dependencies are already installed in Colab and the single missing dependency `s3fs` is first cell (reboot of kernel required after install).  

`03_evaluation.ipynb` is used to load the results from step 1 and 2 and create plots for the report.

For details consult the markdown header and the comments in the notebooks.  