Narrative Knowledge Delta (NKD) for Scientific Articles
=================================================

This repository contains the code, data, prompts and evaluation data for the paper **Benchmark Creation for Narrative Knowledge Delta Extraction Tasks: Can LLMs Help?**.

Structure
------------
The folder structure is based on cookie clutter template, below is the relevant folders for the paper work:

    ├── LICENSE
    ├── README.md          
    ├── data
    │   ├── interviews    <- Contains the _Interviews_ dataset: paper pairs, evaluation data and the corpus used for fact-checking the output from NKD llm.
    │   ├── survey        <- Contains the _Survey_ dataset: paper pairs, evaluation data and the corpus used for fact-checking the output from NKD llm.
    │   ├── synthetic     <- Contains the _Synthetic_ dataset paper pairs.
    │
    ├── notebooks          <- Contains the notebooks and instructions to run: the Multivers fact-check model, and output evaluation.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── prompts        <- Prompts text used in the experiments
    │   │
    │   ├── nkd             <- Contains models to generate the NKD
    │   │   │                
    │   │   ├── llm          <- Contains the scripts to generate NKD based on the pipeline descriped in the paper
    │   │
    │   └── scripts  
    │       └── nkd
    |            └── llm     	<- contains the scripts to accept input and generate NKD by executing the functions in the nkd main directory
    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
