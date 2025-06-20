# Multi-Node Orchestration 

## Introduction
Up until now we've leveraged a single node to do our training. In reality, for training and inference there are times where the model is too large to fit on a single node. Many of the concepts we've introduced up until this point -- batching, paralleism, sharding -- all apply for multiple nodes. But now we need to introdoce a few new concetps: orchestration, storage and networking. 

### Inference vs. Training
With the introduction of multiple nodes, the communication between them and the instatiation of the actual training/inference workloads become more complex. Because of this we'll be splitting into inference/training from this point, as before the concepts generally applied to both of them. But when orchestration is involved, and as the use-cases scale to larger models and higher levels of performance, unique techniques need to be applied. We suggest diving into whichever section is more relevant for your use case. This document will provide an introduction to both, and then split off into scripts/notebooks relevant to the respective use-cases.

## Training
We've covered most of the low level tecniques needed as an introduction to deploying and optimizng a training workload, so from here we will cover orchestration, networking and storage. What is it, how is it utilized, and what are the considerations you need when deploying your workloads.

> Before we get started we encourage you to become familiar with the language we'll be using in the [Appendix](../APPENDIX.md). 


When you're ready read the [Introduction](./training/README.md) and go through the [Labs](./training/module3.trainng.lab1.ipynb).
