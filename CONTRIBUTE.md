# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

## Contribution Guide

### A rough structure is like:

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
