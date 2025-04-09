# DB Cookbook

This is a WIP repo internally for team to work together for each section. Please create your own section folder accordingly. In each section, there are Readme.md and several Jupyter notebooks.

## A rough structure is like:

```
Directory structure:
└── aws-samples-database-cookbook/
    ├── README.md                        //Abstract of the cookbook, table of content for the cookbook, etc. -> this goes to the AWS documentation
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── LICENSE
    ├── RELEASE_NOTES.md
    ├── 01_Getting_Started_with_AWS/
    │   ├── README.md                    //Abstract of the section, table of content for the section, etc. -> this goes to the AWS documentation
    │   ├── sub_section1/
    │   │   ├── notebook_1.ipynb         // Put visual diagrams and architecture, source code with contextual explanation here; If not more than 5 sub steps, put individual gifs of the console experience associated with each sub-step here, otherwise provide a hyperlink to the comprehensive video walkthrough in readme. Include cost, optimization, best pratices, troubleshooting steps associated with the content within this particular notebook here. -> this is linked by the sub-section's readme which goes to the AWS documentation
    │   │   └── README.md                // This is the abstract of the sub-section, table of content for this sub-section, link the notebooks (1, or several) logically here. If there is a comprehensive video walkthrough for all the sub-sections, put it here, otherwise put gifs in the notebooks; If there is a comprehensive visual architecture of this sub-section, put it here. Put CFN for the solution of the sub-section here, otherwise put into the readme of the section; Include cost, optimization, best pratices associated with the sub-section here.  -> this goes to the AWS documentation
    │   ├── sub_section2/
    │   │   ├── notebook_2.ipynb
    │   │   └── README.md                
    │   ├── codes/
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
    Open your video in QuickTime Player .
    Trim the video: to the desired section using Edit > Trim.
    Go to File > Export > Export GIF .
    Choose a location and filename: for the GIF.
    Click "Save" . 

**How to create a screen recording with some basic highlighting functions (e.g. zoom in, spotlight)**
App: Pro Mouse ($3.99)
Demo: https://www.youtube.com/watch?v=LzTkRq8lBqc
If you use Pro Mouse or any App similar to this tool, change the customized color to Amazon orange to be consistent with each other.
You don't have to use Pro Mouse though. As long as the console walkthough/demo is clear and easy for users to understand/follow, it works for the purpose of this cookbook.
