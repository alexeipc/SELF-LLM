# SELF: Self-Extend the Context Length With a Logistic Growth Function

## Overview
SELF uses a logistic growth function to group tokens together in order for LLMs to understand long context prompts better. By grouping the tokens together at far relative distances, LLMs are able to consider these tokens to a high enough degree where they have a meaningful effect on the output. We also ensure that the relatively close tokens are still assigned a greater priority since closer tokens are usually more important. Our method requires no modification to the prompt and has a minimum performance overhead.

## How to use
### Setup environment
Use the docker image [hoytjin/selfextend_docker](https://hub.docker.com/r/hoytjin/selfextend_docker/tags) from LongLM. This will include all packages required for the operation of SELF.  
Also create a directory ``build`` where the program is being executed.

```python
import SelfExtend

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group size, neighbor window. 

SelfExtend.apply(loaded_model, group_size, window_size, enable_flash_attention=False)

# Inference, e.g., loaded_model.generate(...)

```