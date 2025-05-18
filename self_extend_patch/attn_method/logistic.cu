#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

const int MAX_THREAD = 1000;

__device__ double gs_inverse_generating_function(int y, double rate, double capacity) {
    double numerator = std::log(y * capacity - y) - std::log(capacity - y);
    double denominator = rate;
    return (double) numerator / denominator;
}

struct Group {
    int first;
    int last;

    __device__ Group(int first, int last)
        : first(first), last(last) // Use initializer list
    {
    }
};

__global__ void gpu_key_group_id(int n, int capacity, int presum, int last_group_size, Group* groups, int* res) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int id = presum + tid;

    if (id >= n) return;

    int next_group_size = last_group_size + 1;
    int last_group_pos = groups[last_group_size].last;
    int group_id_val = last_group_pos + (id - presum + next_group_size - 1) / next_group_size;

    int max_n = presum + next_group_size * (groups[last_group_size + 1].last - groups[last_group_size + 1].first + 1);
    if (id > max_n && next_group_size != capacity - 1) return;

    if (n - id >= 0 && n >= group_id_val && n - id < n) {
		//printf("id=%d, group_id=%d, n-id=%d\n", id, group_id_val, n - id);
		//printf("%p\n", res);
		//printf("%d\n", res[0]);
	    res[id] = group_id_val;

    }
}

__global__ void gpu_query_group_id(int n, int window_size, int* group_query_position, int* group_key_position) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;

    if (i < window_size) group_query_position[i] = window_size;
    else {
        group_query_position[i] = window_size + group_key_position[i - window_size];
    }
}

__global__ void freq_group(int capacity, double rate, Group* groups) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= 1 && i < capacity - 1) {
		double lower_bound = gs_inverse_generating_function(i, rate, capacity);
        double upper_bound = gs_inverse_generating_function(i + 1, rate, capacity);

		groups[i] = Group(ceil(lower_bound), floor(upper_bound));

		if (upper_bound == (double)floor(upper_bound)) groups[i].last--;
   }
}

void async_generator(torch::Tensor group_query_position, torch::Tensor group_key_position, int n, int window_size, double rate, double capacity) {
	Group* groups;
	int tensor_device = group_key_position.device().index();
cudaSetDevice(tensor_device);
	cudaMallocManaged(&groups, (capacity + 1) * sizeof(Group));
	cudaMemset(groups, 0, (capacity + 1) * sizeof(Group));

	freq_group<<<1, capacity - 1>>>(capacity, rate, groups);
	cudaDeviceSynchronize();


groups[0].last = -1;

    int presum = 0;
	for (int i = 1; i < capacity; ++i) {
        int group_size = groups[i].last - groups[i].first + 1;
        int next_group_size = groups[i + 1].last - groups[i + 1].first + 1;

	if (group_size <= 0 || next_group_size <= 0) continue;
	gpu_key_group_id<<<(n + MAX_THREAD - 1)/MAX_THREAD, MAX_THREAD>>>(n, capacity, presum, i - 1, groups ,group_key_position.data_ptr<int>());
	presum = presum + i * group_size;
    }
	cudaDeviceSynchronize();

    gpu_query_group_id<<<(n + MAX_THREAD - 1)/MAX_THREAD, MAX_THREAD>>>(n, window_size, group_query_position.data_ptr<int>() , group_key_position.data_ptr<int>());
    cudaDeviceSynchronize();
    cudaFree(groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_generator", &async_generator, "Description of your function");
}
