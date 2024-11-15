import torch
import psutil
import time
from prettytable import PrettyTable

class Profiler:
    def __init__(self, model, use_cuda=False):
        self.model = model.cpu().eval()a
        self.use_cuda = use_cuda

        if use_cuda:
            self.model = self.model.cuda()
        
        self.layer_types = {}
        self.layer_times = {}
        self.layer_cpu_memory = {}
        self.layer_gpu_memory = {}
        self.layer_input_shapes = {}
        self.layer_output_shapes = {}
        self.hooks = []
        self.total_time = 0
        self.max_memory = 0
        self.start_time = None

    def register_hooks(self):
        for name, layer in self.model.named_modules():
            self.layer_types[name] = layer.__class__.__name__
            pre_hook = layer.register_forward_pre_hook(self.pre_forward_hook(name))
            post_hook = layer.register_forward_hook(self.post_forward_hook(name))
            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def pre_forward_hook(self, name):
        def hook(layer, input):
            self.start_time = time.time()
            process = psutil.Process()
            current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
            self.layer_cpu_memory[name] = current_cpu_memory
            if isinstance(input, torch.Tensor):
              self.layer_input_shapes[name] = input.shape
            else:
              for inp in input:
                if isinstance(inp, torch.Tensor):
                  self.layer_input_shapes[name] = inp.shape
                  break
                else:
                  pass

            if self.use_cuda:
                torch.cuda.synchronize()
                self.layer_gpu_memory[name] = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        return hook

    def post_forward_hook(self, name):
        def hook(layer, input, output):
            end_time = time.time()
            process = psutil.Process()
            current_cpu_memory = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

            elapsed_time = end_time - self.start_time

            self.layer_times[name] = elapsed_time
            self.layer_cpu_memory[name] = current_cpu_memory
            # self.layer_output_shapes[name] = output.shape if isinstance(output, torch.Tensor) else [out.shape for out in output]
            if isinstance(output, torch.Tensor):
              self.layer_output_shapes[name] = output.shape
            else:
              for out in output:
                # print("here out ", out)
                if isinstance(out, torch.Tensor):
                  self.layer_output_shapes[name] = out.shape
                  break
                else:
                  pass
            if self.use_cuda:
                torch.cuda.synchronize()
                self.layer_gpu_memory[name] = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
            self.max_memory = max(self.max_memory, current_cpu_memory)
        return hook

    def profile(self, input_data):
        if self.use_cuda:
            input_data = input_data.cuda()
        self.register_hooks()
        start_total_time = time.time()
        self.model(input_data)
        self.total_time = time.time() - start_total_time
        self.remove_hooks()

    def print_profiling_info(self, print_io_shape=True):
        table = PrettyTable()
        field_names = ["Row ID", "Layer", "Type", "Time (s)", "CPU Memory (MB)"]
        if self.use_cuda:
            field_names.append("GPU Memory (MB)")
        if print_io_shape:
            field_names.extend(["Input Shape", "Output Shape"])
        table.field_names = field_names

        for i, layer in enumerate(self.layer_times):
            row = [i + 1, layer, self.layer_types[layer], f"{self.layer_times[layer]:.6f}", f"{self.layer_cpu_memory[layer]:.2f}"]
            if self.use_cuda:
                row.append(f"{self.layer_gpu_memory.get(layer, 0):.2f}")
            if print_io_shape:
                o_layers=list(self.layer_output_shapes.keys())
                i_layers=list(self.layer_input_shapes.keys())
                if layer not in o_layers:
                  self.layer_output_shapes[layer]=[]
                if layer not in i_layers:
                  self.layer_input_shapes[layer]=[]
                row.extend([self.layer_input_shapes[layer], self.layer_output_shapes[layer]])
            table.add_row(row)

        print("Layer Profiling Information:")
        print(table)
        print(f"\nTotal Inference Time: {self.total_time:.6f} seconds")
        print(f"Max Memory Consumption: {self.max_memory:.2f} MB")

    def print_top_k_layers(self, k, print_io_shape=True):
        sorted_layers = sorted(self.layer_times.items(), key=lambda x: x[1], reverse=True)
        table = PrettyTable()
        field_names = ["Row ID", "Layer", "Type", "Time (s)"]
        if print_io_shape:
            field_names.extend(["Input Shape", "Output Shape"])
        table.field_names = field_names

        for i, (layer, time_taken) in enumerate(sorted_layers[:k]):
            row_id = list(self.layer_times.keys()).index(layer) + 1
            row = [row_id, layer, self.layer_types[layer], f"{time_taken:.6f}"]
            if print_io_shape:
                row.extend([self.layer_input_shapes[layer], self.layer_output_shapes[layer]])
            table.add_row(row)

        print(f"Top {k} Layers by Execution Time:")
        print(table)
