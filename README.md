# Self-Managed Machine Learning on AWS: A Comprehensive Cookbook (WIP Release)

## Introduction

Welcome to the Self-Managed Machine Learning (SMML) Cookbook for AWS! This guide is designed for ML teams tasked with hosting training and inference workloads for large language models like Llama-Xb. The cookbook provides a structured approach to effectively managing ML workloads at scale while maximizing cost-effectiveness and building a mature platform on AWS.

Whether you're new to self-managed ML or looking to optimize your existing setup, this cookbook will guide you through the complete lifecycle - from understanding model components to deploying high-performance inference endpoints, and everything in between.

## What You'll Learn

* **Model Components**: What assets make up a complete model and how to work with them
* **Hardware Requirements**: Essential knowledge about selecting and configuring appropriate hardware
* **Performance Optimization**: Techniques to extract maximum performance from your infrastructure
* **Workload Distribution**: Methods for distributing ML workloads across single and multiple AWS EC2 instances
* **Scalable Inference**: How to build and manage scalable inference services
* **Comprehensive Monitoring**: How to observe your ML workloads from cluster level down to individual GPU metrics

### Prerequisite Knowledge

Before starting this lab, you should be familiar with:

* [Basic linear algebra (matrices, matrix multiplication)](https://www.mathsisfun.com/algebra/matrix-introduction.html)
* [Python programming and PyTorch fundamentals](https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)
* [Basic understanding of neural network architectures](https://www.ibm.com/think/topics/neural-networks)
* Familiarity with [CUDA](https://blogs.nvidia.com/blog/what-is-cuda-2/) and [GPU](https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html) computing concepts
* Understanding of distributed computing paradigms 

## Journey Overview

This cookbook follows a practical, hands-on approach with foundational concepts applicable to both training and inference, before diving into specific implementation paths:

<p align="center">
   <img src="/assets/roadmap.png" alt="drawing" width="200"/>
</p>

1. **Core ML Infrastructure Concepts** (will do with training but concepts applicable to both training & inference)
   - **Model Components & Mathematics** - Understanding the fundamental building blocks of large language models
   - **Self-Hosting a Model** - Setting up your environment to work with models like Llama
   - **Leveraging Hardware** - Optimizing for specific AWS compute instances and accelerators
   - **Multi-Accelerator Distribution** - Techniques for distributing workloads across multiple GPUs

2. **Multi-Node Orchestration**
   - **Training** - We will discuss the particulars for training and orchestrating multi-node jobs, but part 1 essentially covers this
   - **Inference** - Building robust inference services with optimized request handling, including:
     - LLM Server implementation
     - Distribution frameworks
     - Request handling and optimization
     - Infrastructure considerations

3. **Operational Excellence**
   - **Storage & Networking** - Best practices for model storage, checkpointing, and network optimization
   - **Operations** - Monitoring, observability, error handling, and recovery strategies
   - **Cost Optimization** - Techniques to maximize performance while minimizing costs


## Ready to get started? 
### [Module 0](./00_Getting_Started/Introduction.md) - Getting Started
This provides an introduction to the cookbook as well as your first steps in understanding what components make up a model, how you run a model, and how a model is "compiled."

### [Module 1](./01_Leveraging_Hardware) - Leveraging Hardware
We will run llama-1b on a single GPU, then show the core concepts for optimizing a workload on a *single* gpu using a generic matrix multiplication. We will cover the roofline model, algorithmic intensity and how to fully utilize your accelerators. Then finally apply some of those concepts to llama to see some improvements.

### [Module 2](./02_Multi-Acclerator_Distribution) - Multi-Accelerator Distribution
We will run a larger model (llama-13b) to utilize 4 GPUs. Then we will show base concepts of compute parallelism, data parallelism, scaling laws, distribution frameworks, and finally we'll run llama-13b with some of these concepts in mind.

### [Module 3](./03_Multi-Node_Orchestration/README.md) - Multi-Node Orchestration
In this section we pull back and look at the overarching framework for distributing jobs with an orchestrator. At this point we split between training and inference, if you're interested in an overview on these topics you can start here.

### [Module 4](./04_Cost_Optimization/README.md) - Cost Optimization
Here we learn how to cost optomize our workload by leveraging capacity strategies under our infrastructure.