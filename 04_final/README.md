
## 01_hands_on.ipynb

The hand on notebook was used to get familiar with the chat template and the model architecture of gemma4. We also used this notebook to generate the controll cases for the steering awareness experiment, where we just gave the prompt of the main experiment to gemma without any steering to see if it generates false positives.
The result of the controllcases is saved to `S3` therefore the environment variables `S3_KEY`, `S3_SECRET`, `S3_HOST` should be set.

## 2_1 contrastive vectors

This notebook calculates the contrastive vectors for the 8 prompts from the seed paper and stores it as a safetensor file.
This and later notebooks assumes persistent storage at `/mnt/pv/`..

## 2_2 & 2_3 steering contrastive vectors

This notebook loads the contrastive vectors from the safetensors file for steering (make sure `/mnt/pv/` is mounted and contains the result from 2_1 ).
The 2_2 notebook is good for manual runs and adjustments to the experiment. 2_3 shares the same codebase but the cells are combined and the code is wrapped in a function for repeated calls.
The main loop runns the experiment with combinations of strength x vector x thinking x prefilling. For each configuration 10 responses are sampled and saved to a file.
To run the experiments with the `31B` model choose a sufficiently sized gpu (we observed memory spikes up to 140GB).


## Evaluation 
The main script we used for evaluating the trials with Qwen-as-a-judge is `03_2_judge_batched-ipynb`. Note that this script is designed to be used in modal with additional instructions in the beginning.

We used Qwen-as-a-judge for all evaluations and checked manually for correctness and additional interesting findings.

For both the answer and the Chain of Thought (CoT) seperately, we evaluated the AI models on the final trials, using three metrics: 
- **affirmativity**: Does the model answer YES (it assumes a thought was injected)? *(only if the respective answer / the respective CoT was not prefilled)*
- **identification**: Does the model identify the correct concept that was injected?
- **coherence**: Does the model's response make sense and is it coherent?
Note that for some trials the AI thought for too long and therefore the answer was truncated. In these cases, we only evaluated the CoT.
For all evaluated texts that were prefilled, we explicitly told the judge so and evaluated only identification and coherence, not whether the answer was affirmative.
Additionaly, in all trials with CoT enabled, we evaluated **faithfulness**: Does the model's answer match its CoT? (i.e. if the model says it detected a thought in its CoT, but then answers NO, this is unfaithful)


## 05_analysis.ipynb
In this script we evaluate the judge answers and correct it in some cases where our manual observations disagree.
It assumes a df_evals.json file with all the results.
Here we also generate the plots for the report.




