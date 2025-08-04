# LLM Collective Intelligence

This is the repository for paper **"Extracting Collective Intelligence Factor in LLM Agent Groups"**

## Environment
- Tested OS: Windows
- Python == 3.10.12

## Overview of files

### Experiment on benchmarks

- **./mmlu**  
Code and data for testing on the MMLU-Pro benchmark.  
Usage: python run.py

- **./math**  
Code and data for testing on the MATH benchmark.  
Usage: python run.py

- **./chess**  
Code and data for testing on the Chess benchmark.  
Usage: python run.py

- **./humaneval**  
Code and data for testing on the HumanEval benchmark. This is adapted from https://github.com/openai/human-eval  
Usage: python run.py

- **./commongen**  
Code and data for testing on the Commongen-Hard benchmark.  
Usage: python run.py



### Result analysis

- **results.xlsx**  
Original experiment results, including performance of all LLM agent groups' on all benchmarks.

- **result_analysis.ipynb**  
Code for reproducing all analysis results and Figure 2-7 in the paper.

