# SELF: Self-Extend the Context Length With a Logistic Growth Function

## Overview
SELF uses a logistic growth function to group tokens together in order for LLMs to understand long context prompts better. By grouping the tokens together at far relative distances, LLMs are able to consider these tokens to a high enough degree where they have a meaningful effect on the output. We also ensure that the relatively close tokens are still assigned a greater priority since closer tokens are usually more important. Our method requires no modification to the prompt and has a minimum performance overhead.

## How to use
### Setup environment
Use the docker image [hoytjin/selfextend_docker](https://hub.docker.com/r/hoytjin/selfextend_docker/tags) from LongLM. This will include all packages required for the operation of SELF.  
Also create a directory ``build`` where the program is being executed. \\
If needed, modify the ``SelfExtend.py`` to include an auth token for gated models.

```python
import SelfExtend

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group size, neighbor window. 

SelfExtend.apply(loaded_model, group_size, window_size, enable_flash_attention=False)

# Inference, e.g., loaded_model.generate(...)

```

### Reasoning Models
Reasoning models require a larger max_new_tokens due to the thinking step. 

## Results
| **Model**                                   | Coursera | TOEFL | QuALITY | CodeU | SFiction | Avg. |
|:--------------------------------------------|---------:|------:|--------:|------:|---------:|-----:|
| **Llama-2-7b-chat\***                       | 29.21 | 51.67 | 37.62 | 1.11 | 60.15 | 35.95 |
| SE-Llama-2-7b-chat\*                        | 35.76 | 55.39 | 41.09 | 1.11 | 57.81 | 38.23 |
| SELF-Llama-2-7b-chat\*                      | 36.19 | 56.88 | 41.09 | 0.00 | 60.94 | 39.02 |
| **Llama-2-13b-chat\***                      | 35.75 | 60.96 | 42.57 | 1.11 | 60.15 | 40.11 |
| SE-Llama-2-13b-chat\*                       | 38.95 | 66.17 | 41.09 | 1.11 | 60.15 | 41.49 |
| SELF-Llama-2-13b-chat\*                     | 37.93 | 64.31 | 39.11 | 0.00 | 57.03 | 39.68 |
| **Qwen-7B\***                               | 52.18 | 79.18 | 65.35 | 0.00 | 63.28 | 52.00 |
| SE-Qwen-7B\*                                | 53.20 | 78.07 | 59.41 | 0.00 | 57.03 | 49.54 |
| SELF-Qwen-7B\*                              | 53.34 | 80.67 | 66.83 | 4.44 | 62.50 | 53.56 |
| **Reasoning Model**                         |         |       |        |      |         |      |
| DeepSeek-R1-Distill-Qwen-7B\*               | 58.43 | 66.54 | 48.01 | 2.22 | 60.16 | 47.07 |
| SE-DeepSeek-R1-Distill-Qwen-7B\*            | 54.21 | 66.17 | 40.59 | 6.66 | 62.40 | 45.81 |
| SELF-DeepSeek-R1-Distill-Qwen-7B\*          | 40.27 | 58.74 | 37.50 | 1.11 | 50.78 | 37.68 |
| **Fixed Models**                            |         |       |        |      |         |      |
| Claude 1.3-100k\*                           | 60.03 | 83.64 | 73.76 | 17.77 | 72.65 | 65.97 |
| GPT-4-32k                                   | 75.58 | 84.38 | 82.17 | 25.55 | 74.99 | 73.11 |
| Turbo-16k-0613\*                            | 63.51 | 78.43 | 61.38 | 12.22 | 64.84 | 60.73 |
| ---                                         |         |       |        |      |         |      |
| ChatGLM2-6b-8k\*                            | 43.75 | 53.90 | 40.59 | 2.22 | 54.68 | 34.69 |
| XGen-7b-8k (2k-4k-8k)\*                     | 26.59 | 44.23 | 35.15 | 1.11 | 48.43 | 26.41 |
| ChatGLM2-6b-8k\*                            | 42.15 | 54.64 | 44.05 | 2.22 | 54.68 | 35.95 |
| ChatGLM2-6b-32k\*                           | 47.81 | 55.01 | 45.04 | 2.22 | 57.02 | 39.01 |
| XGen-7b-8k\*                                | 29.06 | 42.37 | 33.66 | 3.33 | 41.40 | 27.63 |
| MPT-7b-65k\*                                | 25.23 | 17.84 | 25.24 | 0.00 | 39.06 | 19.22 |
