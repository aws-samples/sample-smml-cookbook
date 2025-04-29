# Welcome to the Self Managed Machine Learning Cookbook.

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

## A rough structure is like:

```
Directory structure:
└── aws-samples-smml-cookbook/
    ├── README.md                        //Abstract of the cookbook, table of content for the cookbook, etc. -> this goes to the AWS documentation
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── LICENSE
    ├── RELEASE_NOTES.md
    ├── 01_Getting_Started_with_AWS/
    │   ├── README.md                    //Abstract of the section, table of content for the section, etc. -> this goes to the AWS documentation
    │   ├── sub_section1/
    │   │   ├── lab1.ipynb         // Put visual diagrams and architecture, source code with contextual explanation here; If not more than 5 sub steps, put individual gifs of the console experience associated with each sub-step here, otherwise provide a hyperlink to the comprehensive video walkthrough in readme. Include cost, optimization, best pratices, troubleshooting steps associated with the content within this particular notebook here. -> this is linked by the sub-section's readme which goes to the AWS documentation
    │   │   └── README.md                // This is the abstract of the sub-section, table of content for this sub-section, link the notebooks (1, or several) logically here. If there is a comprehensive video walkthrough for all the sub-sections, put it here, otherwise put gifs in the notebooks; If there is a comprehensive visual architecture of this sub-section, put it here. Put CFN for the solution of the sub-section here, otherwise put into the readme of the section; Include cost, optimization, best pratices associated with the sub-section here.  -> this goes to the AWS documentation
    │   ├── sub_section2/
    │   │   ├── lab1.ipynb
    │   │   └── README.md                
    │   ├── src/
    │   │   ├── requirements.txt
    │   │   └── cfn
    │   ├── images/
    │   │   └── img1.gif
    │   └── videos/
    │       └── console1.mp4
...
```

**Note the naming convention of the folders and files**
1. Snake starting with the section or sub-section number: 01_Getting_Started_with_AWS
2. Descriptive

**Note that you can call back the content in other sections/documents and explain visually**
1. Logically link your content within the section and with other sections
2. If you have to explain something by more than 2 sentences, link to where the concept comes from (should be our official documents, blogs, knowledge center articles, workshops, etc. avoid using 3rd documents unless necessary).
3. Use a visual diagram (not necessarily architecture diagram, but also flow chart, or [mind map](https://en.wikipedia.org/wiki/Mind_map) ) to explain logically, instead of put a lecture in the readme.

**Each readme/notebook should NOT be more than 1000 words (exclude codes)**
1. If you feel the readme/notebook is too long, split it out logically.
2. If you have split it out already, but too many sub-sections, eliminate the content that is not critical to our startup developers to launch your solution successfully and link in-depth content in the optimization/best practice section.
3. Revisit our cookbook tenets and FAQs again, and make your judgement call.

**How to create gifs in Mac**

Using QuickTime Player:
1. Open your video in QuickTime Player .
2. Trim the video: to the desired section using Edit > Trim.
3. Go to File > Export > Export GIF .
4. Choose a location and filename: for the GIF.
5. Click "Save" . 

**How to create a screen recording with some basic highlighting functions (e.g. zoom in, spotlight)**

App: Pro Mouse ($3.99)
Demo: https://www.youtube.com/watch?v=LzTkRq8lBqc
If you use Pro Mouse or any App similar to this tool, change the customized color to Amazon orange to be consistent with each other.
You don't have to use Pro Mouse though. As long as the console walkthough/demo is clear and easy for users to understand/follow, it works for the purpose of this cookbook.
