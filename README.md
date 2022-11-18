# hf-compiler

<img src="./assets/gen027.png">

We are trying to build a function index to reduce the amount of duplicate effort (ie. of figuring out the edge cases) that goes into writing code. To do this, we are using a hosted elastic search to index all the files and code snippets and the source data is [the-stack](https://huggingface.co/datasets/bigcode/the-stack-dedup).

## Requirements

- have a running elastic search 

## Steps

- Create an empty file `touch hfcomp/sercret.py` and add the following variables:
  - `HF_TOKEN`: Get them from [here](https://huggingface.co/docs/hub/security-tokens)
  - `ES_URL`: URL to your elastic search
  - `ES_USERNAME`: username for elastic search
  - `ES_PWD`: password for elastic search

> It is assumed that everything is running from the repo root.

- Download and load up the data by running the script `python3 -m hfcomp.downloader`

