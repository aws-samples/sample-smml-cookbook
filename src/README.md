# Welcome to the Self Managed Machine Learning labs.

These labs are meant to demystify the processes of distributing ML workloads for Inference and Training across multiple accelerators and devices. 

We will take on the task of running a model from a very small size (mistral-tiny) to a larger distributed model across multiple instances (mistral-7b).

Throughout this workshop we will begin by deploying a simple version of the model we'll explore, then establish concepts using generic matrix multiplications. We do this because *most* modern machine learning algorithms consist almost entirely of matrix multiplications. Once you understand the concept of how a matrix multiplication is optimize, you should be able to easily transfer that understanding to the models you run.

Although we are using a real-life example, we will also dive into the concepts and science behind optimizing these workloads, in a way that ideally will establish a foundation for you to apply these concepts to larger models and larger clusters.

Here's an overview of the labs:

## [Lab 0](./lab0.ipynb)
We will run the model, work through pulling it down and a basic deployment as well as basic benchmarking. This will serve as a foundation for the use case we'll work through. This can be skipped if you are familiar with these concepts.

## [Lab 1](./lab1.ipynb)
We will run mistral-tiny on a single GPU, then show the core concepts for optimizing a workload on a *single* gpu using a generic matrix multiplication. We will cover the roofline model, algorithmic intensity and how to fully utilize your accelerators. Then finally apply some of those concepts to mistral-tiny to see some improvements.

## [Lab 2](./lab2.ipynb)
We will run a larger model (mistral-7b) to utilize 4 GPUs. Then we will show base concepts of compute parallelism, data parallelism, scaling laws, distribution frameworks, and finally we'll run mistral-7b with some of these concepts in mind.

## [Lab 3](./lab3.ipynb)
In this Lab we'll utilize a cluster to distribute a model across 8 GPUs, and 2 nodes. We'll cover job orchestrators, networking overhead, and collective communications. 

## [Lab 4](./lab4.ipynb)
TBD Performance