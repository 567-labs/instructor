# Introduction

This is a simple example which shows how to perform Chain Of Density summarization using GPT-3.5 and utilise the generated output to fine-tune a 3.5 model for production usage. All of our data referenced in this file is located [here](https://huggingface.co/datasets/ivanleomk/gpt4-chain-of-density) on hugging face

Check out our blog post [here](https://jxnl.github.io/instructor/blog/2023/11/05/implementing-chain-of-density/) where we have a detailed explanation of the code and a [colab notebook](https://colab.research.google.com/drive/1iBkrEh2G5U8yh8RmI8EkWxjLq6zIIuVm?usp=sharing) walking you through how we perform our calculations.

## Instructions

1. First, install all of the required dependencies by running the command below. We recommend using a virtual environment to install these so that it does not affect your system installation.

> We use NLTK to ensure that our summaries are of a certain token length. In order to do so, you'll need to download the `punkt` package to compute the token metrics. You can do so by running the command `nltk.download('punkt')`

```
pip3 install -r requirements.txt
```

2. Download the `test.csv` file and the `summarization.jsonl` file that you want to use for finetuning. We provide one with `20` examples, `50` examples and `100` examples to be used for testing. Let's now run a simple finetuning job with the following command.

> Don't forget to set your `OPENAI_API_KEY` as an environment variable in your shell before running these commands

```
instructor jobs create-from-file summarization.jsonl 
```

3. Once the job is complete, you'll end up with a new GPT 3.5 model that's capable of producing high quality summaries with a high entity density. You can run it by simply changing our `finetune.py` file's `instructions.distil` annotator as

```
@instructions.distil(model=<your finetuned model >,mode="dispatch")
def distil_summarization(text: str) -> GeneratedSummary:
// rest of code goes here
```