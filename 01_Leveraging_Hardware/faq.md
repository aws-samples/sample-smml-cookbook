# GPU Optimization Workshop FAQ

## General Workshop Questions

### Q: What's the distinction between accelerators and GPUs in your introduction?
A: The introduction mentions both terms as they're sometimes used interchangeably in ML contexts. For this workshop, we're specifically focusing on GPUs. However, the optimization techniques discussed can apply to other accelerators like Trainium, Inferentia, TPUs or specialized ML hardware. For simplicity, you can think of GPUs as one type of accelerator.

### Q: How long should I spend on the pre-requisite readings?
A: Plan for approximately 30-45 minutes to review the roofline model concept and about 15-20 minutes for GEMM operations if you're new to these topics. Basic GPU hardware knowledge might require another 15-30 minutes if you're unfamiliar with GPU architecture.

### Q: The Matrix Multiplication resource was difficult to understand. Are there alternatives?
A: We recommend these more accessible resources for beginners:
- [This Matrix Multiplication Intro](https://www.youtube.com/watch?v=XkY2DOUCWMU) by 3Blue1Brown on YouTube
- "Understanding GEMM Operations in Deep Learning" from the NVIDIA Developer Blog
The original resource is still valuable for those wanting a deeper mathematical understanding.

In general we encourage you to dive into perriferal topics you're unsure of. This complex topic, but complexitiy comes from the many cross-displines. Any individual topic can be learned at a surface level very quickly.

### Q: How do I interpret the the Roofline Model Charts?

A: The roofline plot visualizes your workload's performance relative to hardware limits. Here's how to interpret these important elements:

**The Dots:**
- "WS" stands for "World Size" (number of GPUs) and "B" indicates batch size
- Each dot represents a specific configuration's performance measurement
- Format: `WS=1,B=X,D=4096,F=4096,L=5` where:
  - WS=1: Using one GPU
  - B=X: Batch size (varies in different runs)
  - D/F: Matrix dimensions for the workload
  - L: Number of layers

**How the Dots Are Calculated:**
1. **Horizontal position (x-axis)**: Represents Arithmetic Intensity in FLOPs/Byte - calculated as:
   - Total floating point operations / Total memory bytes accessed
   - Higher values mean more computation per memory access (more efficient)

2. **Vertical position (y-axis)**: Represents achieved performance in TFLOP/s - calculated as:
   - Total operations performed / Total execution time
   - Higher values mean better computational throughput

**What to Look For:**
- **Dots on the sloped portion** (left side of green line): These configurations are memory-bandwidth limited
- **Dots on the flat portion** (right side of green line): These configurations are compute-limited
- **Dots farther up the slope**: Achieving better hardware utilization
- **Movement from left to right** as batch size increases: Shows how batching improves arithmetic intensity

**The Lines:**
- **Orange sloped line**: Maximum performance possible at each arithmetic intensity
- **Red horizontal line** (121.0 TFLOPS): Maximum compute capability of the GPU
- **Green vertical line** (403.33 FLOPs/Byte): The "ridge point" where the bottleneck shifts from memory to compute

**Practical Interpretation:**
In the third graph, you can clearly see how increasing batch size moves the dots rightward and upward, showing improved GPU utilization. The goal is to reach configurations that place your workload as close to the "roof" as possible, maximizing your hardware investment.

This visualization demonstrates why batching works: it increases arithmetic intensity by reusing weights for multiple inputs, allowing you to perform more computations per memory access and approach your GPU's theoretical peak performance.

## Technical Setup Questions

### Q: More comments about each included library in the imports section
A: Here's additional context for some libaries we use:
- `torch`: PyTorch deep learning framework
- `pynvml`: NVIDIA Management Library for GPU metrics
- `matplotlib`: Plotting and visualization
- `transformers`: Hugging Face library for pre-trained models

### Q: Why are there 4 GPUs listed when we're only focusing on one?
A: We show all available GPUs in the environment to provide context about the hardware setup. We only use the first GPU (index 0) for this lab, but displaying all available GPUs helps participants understand the complete environment they're working in. This becomes important in the next lab where we expand to multiple GPUs.

### Q: I'm getting errors like "missing required parameter 'tok'" or "src.utils.model_utils not found" after restarting my notebook. How do I fix this?

A: These errors typically occur because Jupyter notebooks don't maintain variable state between kernel restarts or when cells are run out of order. Here's how to troubleshoot:

**Common Issues:**
- Missing variables (like the `tok` tokenizer)
- Import errors for custom modules
- "Name not defined" errors for previously created objects

**Step-by-Step Fix:**
1. **Restart from the top**: After a kernel restart, re-run cells sequentially from the beginning
2. **Check directory setup**: Ensure the `os.chdir(parent_dir)` cell has run to set the correct working directory
3. **Verify imports**: Make sure all import statements have executed successfully, especially those with `importlib.reload()`
4. **Check for dependencies**: Some objects depend on others (e.g., the tokenizer `tok` must exist before using it in `benchmark_llm()`)
5. **Inspect variable state**: Use `%whos` magic command to list all variables in the current session

**Best Practices:**
- Create helper functions that initialize necessary objects if they don't exist
- Add defensive checks in your code (e.g., `if 'tok' not in globals(): tok = AutoTokenizer.from_pretrained(model_name)`)
- Consider using a `setup()` function that handles all initialization in one place
- Add cell tags or comments indicating dependencies between cells

When in doubt, the safest approach is always to restart the kernel and run all cells in order from the beginning.

## Model Configuration Questions

### Q: Can you provide more context for flash attention and xformers flags?
A: Flash Attention is an optimized attention implementation that significantly speeds up transformer models. In this lab, we're using "eager" implementation as a baseline. In more advanced scenarios, you might switch to flash attention or xformers for better performance. These alternatives provide more efficient memory usage and faster computation for attention mechanisms.

This is an advanced concept, we just apply it so we can get the model running appropriately.

### Q: What is the KV-cache?
A: The KV-cache (Key-Value cache) stores the attention keys and values from previous tokens in transformer models to avoid recomputing them. By setting `model.config.use_cache = False`, we disable this feature for training to ensure we compute gradients properly. For inference, enabling the KV-cache would improve performance.

This is an advanced concept, we just apply it so we can get the model running appropriately.

### Q: What is eager PyTorch?
A: "Eager" refers to PyTorch's default execution mode where operations are executed immediately when defined. This contrasts with "graph" mode (like TorchScript or torch.compile), which can optimize execution by analyzing the entire computational graph. We use eager mode here for simplicity and transparency.

### Q: Why do we disable cudnn/TF32 matmul fusing?
A: We disable these optimizations (`torch.backends.cudnn.enabled = False` and `torch.backends.cuda.matmul.allow_tf32 = False`) to establish a clean baseline. These features provide automatic optimizations that might mask the raw performance differences we want to demonstrate with batching. In production, you would typically leave these enabled.

## Execution and Troubleshooting Questions

### Q: I got "TypeError: benchmark_llm() missing 1 required positional argument: 'tok'". How do I fix this?
A: The `benchmark_llm()` function requires a tokenizer as its `tok` parameter. Make sure you're passing the tokenizer you created earlier:

\```python
elapsed_s, tokens_generated, _, _ = mutils.benchmark_llm(model, tok,
                         batch=batch, seq_len=seq_len)
\```

The tokenizer should have an `eos_token` attribute, which is typically set when you load it from the model hub.

### Q: What should I look for when running `watch nvidia-smi`?
A: When running `watch nvidia-smi` in a terminal while executing GPU operations, look for:
- GPU utilization percentage (higher = better GPU usage)
- Memory usage (increasing as batch sizes grow)
- Power consumption (indicates computational intensity)
- Temperature (to ensure the GPU isn't overheating)

These metrics provide real-time feedback on how effectively your code is utilizing the GPU.

## Understanding Results Questions

### Q: What does the blue dot labeled "WS" represent in the roofline models?
A: The "WS" (World Size) dot represents your current workload's position on the roofline model. Its horizontal position shows the arithmetic intensity (operations per byte), and its vertical position shows the achieved FLOPS. As batch size increases, this dot moves to the right and upward, indicating better GPU utilization.

### Q: How do you determine the right batch size beyond trial and error?
A: While experimentation is important, you can use these guidelines:
1. Start with the roofline model to identify your compute or memory bottlenecks
2. Calculate memory requirements: `batch_size * sequence_length * hidden_size * bytes_per_element < available_GPU_memory`
3. Monitor diminishing returns - when increasing batch size no longer significantly improves FLOPS or reduces cost
4. Consider latency requirements for your application
5. For training, gradient accumulation can simulate larger batch sizes

The optimal batch size depends on your specific model architecture, GPU hardware, and application requirements.

### Q: Can you define sequence length and other terms in an appendix?
A: Yes, here are some key terms:
- **Sequence Length**: The number of tokens in each input sample
- **Batch Size**: The number of samples processed simultaneously
- **Arithmetic Intensity**: The ratio of compute operations to memory operations (FLOP/Byte)
- **FLOPS**: Floating Point Operations Per Second, measuring computational throughput
- **Roofline Model**: Performance model showing the relationship between arithmetic intensity and achievable performance

### Q: How do I incorporate this into tangible work?
A: When deploying an actual model use this knowledge to consider:
1. Performance requirements (latency vs. throughput)
2. Discuss whether your use case benefits from batching (e.g., batch inference can significantly reduce costs)
3. Explain how hardware selection should consider both memory bandwidth and compute capability
4. Recommend profiling the specific workload to identify bottlenecks
5. Show cost implications with concrete examples (like our batch size vs. cost chart)

Rather than asking "have you run a rooftop analysis?", frame it as "Let's look at how we can optimize your model's performance-to-cost ratio with batching techniques."

## Additional Questions

### Q: Will these techniques work for all models?
A: These principles apply broadly, but the specific gains vary by model architecture. Transformer models like Llama benefit significantly from batching due to their matrix multiplication-heavy operations. CNNs and other architectures may have different optimal batch size sweet spots.

### Q: How do these optimizations relate to the next lab on multi-GPU deployments?
A: This lab focuses on optimizing a single GPU before scaling to multiple GPUs. The batching concepts and roofline analysis provide a foundation for understanding distributed deployment. In multi-GPU scenarios, we'll build on these concepts while introducing new considerations like communication overhead between GPUs.