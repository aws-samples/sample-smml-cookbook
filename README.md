# Welcome to the Self Managed Machine Learning Cookbook.

These labs are meant to demystify the processes of distributing ML workloads for Inference and Training across multiple accelerators and devices. 

We will take on the task of running a model from a very small size (llama) to a larger distributed model across multiple instances (llama-13b).

Throughout this workshop we will begin by deploying a simple version of the model we'll explore, then establish concepts using generic matrix multiplications. We do this because *most* modern machine learning algorithms consist almost entirely of matrix multiplications. Once you understand the concept of how a matrix multiplication is optimize, you should be able to easily transfer that understanding to the models you run.

Although we are using a real-life example, we will also dive into the concepts and science behind optimizing these workloads, in a way that ideally will establish a foundation for you to apply these concepts to larger models and larger clusters.

Here's an overview of the labs:

## [Module 1](./01_Leveraging_Hardware)
We will run llama-1b on a single GPU, then show the core concepts for optimizing a workload on a *single* gpu using a generic matrix multiplication. We will cover the roofline model, algorithmic intensity and how to fully utilize your accelerators. Then finally apply some of those concepts to llama to see some improvements.

## [Module 2](./02_Multi-Acclerator_Distribution)
We will run a larger model (llama-13b) to utilize 4 GPUs. Then we will show base concepts of compute parallelism, data parallelism, scaling laws, distribution frameworks, and finally we'll run llama-13b with some of these concepts in mind.
