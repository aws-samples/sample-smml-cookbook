
# Getting Started
Before we jump into any code we want to level set the environment and language we intent to utilize throughout the cookbook.

## Llama Model Architecture Overview

While this cookbook uses Llama as our reference model, the principles apply to most large language models. To help you understand key terminology:

* **Attention Heads**: Components that allow the model to focus on different parts of the input sequence
* **Context/Sequence Length**: The maximum number of tokens a model can process at once
* **Model Parameters**: The learnable weights that define the model's behavior (measured in billions)
* **Tokenization**: The process of converting text into numerical representations
* **Transformer Architecture**: The foundational architecture behind modern LLMs

For a deeper dive into Llama's architecture, refer to the [SMML GitHub](https://github.com/example/smml) resources.

## Hardware Environment

This cookbook focuses on AWS g6.12xlarge instances equipped with NVIDIA L4 GPUs for demonstration purposes. While production environments would likely use larger instances with more powerful GPUs or specialized accelerators, the concepts presented apply broadly to various hardware configurations:

* Single-instance optimization for both training and inference
* Multi-instance cluster deployment for distributed workloads
* Network and storage configurations optimized for ML

All principles covered can be applied to larger GPUs (like A10G, A100, H100) and different AWS accelerators (like Trainium and Inferentia). One notable difference between our demonstration setup and some production configurations is the absence of high-speed GPU interconnects like NVLink, which we'll specifically address in the relevant sections on multi-accelerator distribution and training.

> Note that while we'll discuss how to deploy and run workloads on these clusters, the cookbook links to external guides for the initial cluster setup.

## Why Learn this Content
<ToDo>
- As you make higher level decisions this is why you see the behavior that you do
- Unerstand conversations with customers in depth

## Key Resources

<!-- * 🧪 [SMML Workshop](https://workshop.example.com/smml) - Hands-on labs and exercises -->
<!-- * 📚 [SMML Cookbook](https://cookbook.example.com/smml) - Detailed guides and recipes -->
* 🗺️ [SMML Roadmap](https://roadmap.sh/r/self-managed-machine-learning) - Visual roadmap version of the cookbook

> We encourage you to utilize the [Appendix](../APPENDIX.md) provided for diagrams and definitons we use throughout.

## Let's get started
To begin your self-managed ML journey:

1. Clone the SMML GitHub repository to access code examples
2. Read along or deploy a similar enviornment to run the code yourself

Ready? Let's begin by exploring the components that make up a [language model like Llama](./module1.lab1.ipynb)...
