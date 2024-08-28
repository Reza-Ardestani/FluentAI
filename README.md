## FluentAI
Disfluency Correction in Questions 

<strong>⚠️ Please read the report in the documentation folder for  comprehensive analysis of the result, research steps, and development phases.</strong>

## Table of Contents

- [Development Summary](#development-summary)
    - [Sprint 1](#sprint-1)
    - [Sprint 2](#sprint-2)
    - [Results Summary](#results-summary)
    - [Future Works](#future-works)
- [Running Models](#running-models)
- [Replicating Results](#replicating-results)
- [References](#references)
- [Acknowledgement](#acknowledgement)




## Development Summary
![Machine Learning Patterns and Principles](https://raw.githubusercontent.com/Reza-Ardestani/ConversationalAI/main/2_RAG_KnowledgeGraph_inventures_CaseStudy/2.0_data/ML_Patterns_Principles_big_pic.jpg)

Following my Machine Learning Desing Patterns and Principles pipeline, Presented at Google Developer Groups, I reached near human performance on this task, disfluecny correction, within two rapid development cycles. In the follwoing subsections, I will present these two cycles, demostrate the result, and list future works. 
<br>

### Sprint 1
I read the original paper of the dataset [1] to gain deeper understanding of the <strong>learning task</strong> and the <strong>benckmark dataset</strong>. Regarding evaluation metrics, I researched about the details of the suggested <strong>metrics</strong> (BLEU, and GLEU) to assess their alignment with the learning task. Finally, I identified two <strong>model</strong> technique, - classification and seq2seq generation - for correcting disfluencies. I opted the latter technique and selected an appropeate <strong>loss function</strong>.

The following picture from my HippoRAG tutorial [2] illustrates this chain of task to modeling that should be carefully aligned.
![Task to Model Chain](./task_to_model_chain.png)

Here are the highlights of the first sprint:
<li>Researched the most suitable models and techniques for each part of the ML pipeline.</li>
<li>Developed a model and evaluated it based on BLEU and GLEU scores.</li>
<li>Thematically analyzed 100 records with the lowest GLEU scores.</li>
<li>Reviewed cases where the model performed very well to explain the reasons.</li>
<li>Found and filled missing values in the train and dev datasets using the GPT-4 model and prompt engineering.</li>
<li>Performed several Exploratory Data Analyses (EDA) to find the best values for hyperparameters, such as max input length.</li>


### Sprint 2

Here are the highlights of the second sprint:
<li>Implemented model confidence scores for detecting out-of-distribution data or anomalies during inference/production.</li>
<li>Implemented Confidence Intervals using Bootstrap resampling, allowing for robust estimation of uncertainty.</li>
<li>Studied state-of-the-art semantic embedding models and used the NomicAI model to calculate a semantic similarity metric, which correlates better with human judgment on the fluency of questions.</li>
<li>Developed version 2.1 of the model and used early stopping to avoid the overfitting problem.</li>
<li>Developed version 2.2 of the model, which outperformed version 1.1 while using even fewer beams.</li>
<li>Identified promising future directions for improving the model's performance and robustness.</li>

### Results Summary
The following table, demostrates the performance of my models on the original test dataset. Average metric scores are obtained by 1000 replicates and ± is followed by an estimated margin of error under 95\% confidence. 
<br>
| Model            | B1            | B2            | B3            | B4            | GLEU          | Semantic Sim | Model Confidence    |
|---------------|---------------|---------------|---------------|---------------|---------------|--------------|---------------|
| V1.1            | 95.58 ±0.37   | 91.64 ±0.59   | 88.78 ±0.72   | 86.35 ±0.89   | 0.90 ±0.01    | 0.98 ±0.00   | 0.96 ±0.00    |
| v2.2            | 95.52 ±0.40   | 91.69 ±0.60   | 88.90 ±0.78   | 86.62 ±0.82   | 0.90 ±0.01    | 0.98 ±0.00   | 0.92 ±0.01    |

<br>
Model version 2.2 achieves better performance on most metrics by using fewer beams (V1.1 uses 8 beams, while V2.2 uses 4 beams) and runs twice as fast (V1.1 takes 1 hour, while V2.2 takes 30 minutes, both tested on a T4 GPU).
<br>Note: Higher model confidence does not necessarily indicate a better model.</br>



### Future Works

<li>Conducted a manual disfluency correction on a sample of test records to establish human-level performance on each metric.</li>
<li>Performed data augmentation with a focus on coreference resolution.</li>
<li>Used model confidence to flag anomalies or out-of-distribution data at inference time.</li>
<li>Implemented a secondary loss function to help the model consider semantic similarity.</li>


## Running Models (demo.py)

Please download the models' weights from:<br>
[Model V1.1](https://drive.google.com/file/d/1-1HTcfDnM532p9_GNLauD3VPtC3z1Huq/view?usp=share_link)
<br>
[Model V2.2](https://drive.google.com/file/d/1VNHy70c_-6kerjvxHb__lPIE_cfq_wug/view?usp=share_link)
<br>

Alternatively, If you are on Google Colab, you can download the models as follows:

```shell
!gdown --id "1VNHy70c_-6kerjvxHb__lPIE_cfq_wug" # V2.2 (Best) model
!gdown --id "1-1HTcfDnM532p9_GNLauD3VPtC3z1Huq" # V1.1 model
!unzip file_name.zip # Then Unzip the model
```

If you are on google colab, you don't need to install any library. Otherwise, install requirements.txt file.

<b>After changing the checkpoint_address to the model's folder,</b> you are able to run the code to correct disfluencies.

```python
import torch
import argparse
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def correct_disfluency(disfluent_question, tokenizer, model):
  """
  Function to generate fluent question from disfluent input
  """
  inputs = tokenizer(disfluent_question, return_tensors="pt", max_length=64,truncation=True).to(device)
  outputs = model.generate(inputs["input_ids"], min_length=3, max_length=64,
                           num_beams=4, no_repeat_ngram_size=3, 
                           repetition_penalty=1.1, early_stopping=True,)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
  running_on_colab = True
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_address', type=str, default='FluentAI_v2.2_best_model' )
  parser.add_argument('--disfluent_input', type=str, default="Disfluent" )

  if running_on_colab:
    # if running the code on google colab
    args = parser.parse_args(args=[])
    # example Where is the closest coffee shop sorry gym?
    args.disfluent_input = "Your Disfluent Question Here" 
  else:
    args = parser.parse_args()

  tokenizer = BartTokenizer.from_pretrained(args.checkpoint_address)
  model = BartForConditionalGeneration.from_pretrained(args.checkpoint_address).to(device)
  print(correct_disfluency(args.disfluent_input, tokenizer, model))
```


## Replicating Results

<B>Download models and setup the environment:</B><br>
We need to set up the environment and install necessary packages. If you are on google colab, run the following script to install the necessary files. <b>Change the runtime to GPU T4 as well.</b> 

```shell
!pip install datasets
!pip install sentence_transformers
!pip install gensim==4.3.2 # GLEU score
!pip install sacrebleu # BLEU score
!pip install nltk # word tokenization
!pip install --upgrade nltk
```

If you want to run the model on your GPU servers, you need to install requirements.txt file. Please contact me in case you need help with setting up GPU and CUDA on your servers.

<B>Download datasets:</B><br>

To replicate model V1.1 download the _"data/origianl"_ folder and for model v2.2 download _"data/filled_missings_of_train_and_dev"_. Keep the prefixes for dataset files (Disfl_QA_*.json).


<B>Fine-tuning:</B><br>
Having setup the environment, downloaded the datafolder, we can fine-tune the BART-large model for replicating my result. The only thing we need to do is passing arguments as follows:


<b> Replicating model V1.1:</b>
```shell
fluentAI.py --checkpoint_address facebook/bart-large --tokenizer_address facebook/bart-large --batch_size 4 --max_epochs 3 --num_beams 8 --raw_data_address ../data/original --job_name FluentAI_v1.1 --train --evaluate
```

<b> Replicating model V2.2:</b>
```shell
fluentAI.py --checkpoint_address facebook/bart-large --tokenizer_address facebook/bart-large --batch_size 4 --max_epochs 5 --num_beams 4 --raw_data_address ../data/filled_missings_of_train_and_dev --job_name FluentAI_v2.2 --train --evaluate
```

If you are running on Google Colab, pass the arguments in a list according to the following example. You can pass more arguments if needed.

```python
args = prepare_args(running_on_notebook=True, args=["--checkpoint_address", "facebook/bart-large", "--tokenizer_address", "facebook/bart-large", "--raw_data_address", "../data/filled_missings_of_train_and_dev", "--job_name", "FluentAI_v2.2", "--train" ,"--evaluate"])
```


## References

The complete list of references are in the report file.
<br><br>
[1] DISFL-QA: A Benchmark Dataset for Understanding Disfluencies in Question Answering

[2] My toturial: [https://youtu.be/wIkIxrsnUs8?si=3NtkXKgC87_Y7w2H&t=1515](https://youtu.be/wIkIxrsnUs8?si=3NtkXKgC87_Y7w2H&t=1515)


## Acknowledgement

I would like to acknowledge the use of Generative AI for assisting with grammar checks in my report and for debugging my code.

<br>
<br>
<br>

### Thank you for your interest in my AI development

