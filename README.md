```python
import SelfExtend

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group size, neighbor window. 

SelfExtend.apply(loaded_model, group_size, window_size, enable_flash_attention=False)

# Inference, e.g., loaded_model.generate(...)

```