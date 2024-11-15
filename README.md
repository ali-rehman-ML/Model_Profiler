# Model_Profiler
The Model Profiler tool provides detailed profiling of PyTorch models, focusing on layer-by-layer execution time and memory usage. It incorporates both high-level and low-level profiling information to analyze model performance.
**Output Table Content Explanation:**
- **Row ID:** Sequential identifier for each layer.
- **Layer:** Name of the layer in the PyTorch model.
- **Type:** Type of layer (e.g., Conv2d, BatchNorm2d).
- **Time (s):** Execution time of the layer during inference.
- **Memory (MB):** Current occupied  CPU memory.
- **Memory (MB):** Current occupied  GPU memory.
- **Input Shape:** Layer Input and Output shape.
- **Output Shape:** Layer Output and Output shape.



### Currently Supported framework :
  - **Pytorch**
  - **ONNX**



### Usage

**Using Memory Profiler (pyTorch):**

   ```python
   from ModelProfiler import Profiler
   import torch
   import torchvision.models as models

   # Define your model and input data (be carefull make sure you use right input size for your model otherwise you may encounter error)
   model = models.resnet50(pretrained=True)
   model.cpu().eval()
   input_data=torch.randn(1, 3, 512, 512).cpu()

   # Create a profiler instance
   profiler = Profiler(model,use_cuda=False)

   # Profile the model with input data
   profiler.profile(input_data)

   # Print detailed profiling information
   profiler.print_profiling_info(print_io_shape=True)

   # Prompt user to input K for top K slowest layers
   k = 10
   
   # Print top K layers by execution time
   profiler.print_top_k_layers(k)

   ```
**Using Memory Profiler (ONNX):**

   ```python
   from ModelProfiler import Profiler


   # Define your model and input data (be carefull make sure you use right input size for your model otherwise you may encounter error)
   model_path = 'model.onnx'

   # Create a profiler instance


   profiler = Profiler(model=model_file,type='onnx',intra_op_num_threads=6,export_txt=True)

   # Profile the model with input data
   profiler.profile(input_data)

   # Print detailed profiling information
   profiler.print_profiling_info(print_io_shape=True)

   # Prompt user to input K for top K slowest layers
   k = 10
   
   # Print top K layers by execution time
   profiler.print_top_k_layers(k)

   ```
