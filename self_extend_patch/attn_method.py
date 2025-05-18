import torch
import math
from torch.utils.cpp_extension import load

async_generator_module = load(
    name="async_generator", 
    sources=["self_extend_patch/attn_method/logistic.cu"], 
    build_directory="build",
    verbose=True
)

async_generator_qlen_1_module = load(
    name="async_generator", 
    sources=["self_extend_patch/attn_method/logistic_qlen_1.cu"], 
    build_directory="build",
    verbose=True
)

print("Load module successfully")
def generate_logistically_grouping_position(q_max, window_size, rate = 0.02, capacity=33, device="cuda", qlen_1 = False):
    if not qlen_1:
        group_query_position = torch.zeros(q_max, dtype=torch.int32, device=device)  
        group_key_position = torch.zeros(q_max, dtype=torch.int32, device=device)
        
        async_generator_module.async_generator(group_query_position, group_key_position, q_max, window_size, rate, capacity)

        group_query_position = group_query_position.unsqueeze(0)
        group_key_position = group_key_position.unsqueeze(0) 
        
        return group_query_position, group_key_position
    else:
        group_key_position = torch.zeros(q_max, dtype=torch.int32, device=device)
        async_generator_qlen_1_module.async_generator(group_key_position, q_max, window_size, rate, capacity)
        group_key_position = group_key_position.unsqueeze(0) 
        
        return group_key_position
