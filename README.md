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
