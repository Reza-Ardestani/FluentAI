"""
* @note: Running the seoncd round of my agile development on Colab (Version 2.1, 2.2)
* @date: Aug 26
* @note: literature, implementing an early version, developing better evaluation, gaining insight for the result for future improvement (Version 1)
* @date: Aug 25
* @Author: Reza Ardestani (ardestani.reza@proton.me)
*
"""

##  1.1) Importing Libraries

## Essential libs
import os
import pickle
import time
import datetime
import warnings
from math import ceil
import numpy as np
import logging
import argparse
import functools
import json
import gc # garbage collector

## Evaluation and visualization libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import sacrebleu
import matplotlib.pyplot as plt

## For tracking ram & gpu
import subprocess
import psutil

## libs for modeling and embeddings
import torch
import torch.nn.functional as F # Using softmax from F
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer # for nomic AI embeddigns
import torch.distributed as dist
import torch.utils.data.distributed



## 1.2) Setting up args and a few global vars
# Args
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def prepare_args(running_on_notebook=False,args=None):
  """
  running_on_notebook: If you are running this python script file from
  commandline standalone, argparse can grab the arguments from the CLI.
  But, if you are using notebooks, like google colab, the environment
  automatically passes some arguments when running a notebook, which can
  conflict with argparse expecting command-line arguments. To work arround
  this issue I have this on_notebook flag

  args: in case running_on_notebook, you should pass arguments in a list.
  like, ['--lr', '0.01', '--batch_size', '512']. if all the args are
  optional and want to use the default values you should pass an empty list.

  Having these two params makes our code versatile.
  """

  # Create parser object
  parser = argparse.ArgumentParser(description='ArgParser')


  # Add arguments
  ngpus_per_node = torch.cuda.device_count()

  """ This next line is the key to getting DistributedDataParallel working on SLURM:
  SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the
  current process inside a node and is also 0 or 1 in this example."""
  # local_rank = int(os.environ.get("SLURM_LOCALID"))
  # rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
  local_rank, rank = 0, 0

  parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
  parser.add_argument('--dist-backend', default='gloo', type=str, help='')
  parser.add_argument('--world_size', default=1, type=int, help='')
  parser.add_argument('--distributed', action='store_true', help='')
  parser.add_argument('--num_workers', type=int, default=1, help='')
  parser.add_argument('--rank', type=int, default=rank, help='Global rank of the current GPU')
  parser.add_argument('--local_rank', type=int, default=local_rank, help='Local rank of the current GPU')
  parser.add_argument('--ngpus_per_node', default=ngpus_per_node, type=int,
                      help='0 means use cpu, 1 means there is no parallel ' + \
                      'computation, more than one means we can use parrallel computing.' + \
                      "Note that on ComputeCanada clusters you can use " + \
                      "parrallel setup w/ 1 gpu but not on GoogleColab.")

  parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')
  parser.add_argument('--batch_size', type=int, default=20, help='')
  parser.add_argument('--max_epochs', type=int, default=5, help='')
  parser.add_argument('--num_beams', type=int, default=4, help='')
  parser.add_argument('--repetition_penalty', type=float, default=2.0, help='')
  parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help='')
  parser.add_argument('--max_encoder_input_lenght', default=64, type=int, help='Tokenizer maximum length.')
  parser.add_argument('--max_decoder_output_lenght', default=64, type=int, help='Generation maximum length.')
  parser.add_argument('--min_decoder_output_lenght', default=4, type=int, help='Generation minimum length.')
  parser.add_argument('--job_name', default='FluentAI_v1.1', type=str, help='Used for saving the artifact, like log, with unique name.')
  parser.add_argument('--checkpoint_address', default="./disfluency_bart_model_v1.1", type=str, help='Checkpoint address containing model_states, optimizer_states, etc.')
  parser.add_argument('--tokenizer_address', default='./disfluency_bart_model_v1.1', type=str, help='')
  parser.add_argument('--dataset_objects_address', default=None, type=str,
                      help='folder address of the pickled dataset objects.'+ \
                      'Inside this folder we expect to have three pkl files of format <X_dataset.pkl> where X is train or test or val.')
  parser.add_argument('--raw_data_address', default='./DISFL_raw_data', type=str,
                      help='relative folder address of the raw DISFL dataset.')
  parser.add_argument("--debug",action='store_true',)
  parser.add_argument("--train",action='store_true',)
  parser.add_argument("--evaluate",action='store_true',)

  # Parse args
  if running_on_notebook and args is not None:
    # Parse from a list manually created
    args = parser.parse_args(args=args)
  else:
    # Parse from the CLI automatically
    args = parser.parse_args()

  # Return args to use the args anywhere in the code like> lr_rate = args.lr
  return args


def print_args_in_logs(myargs):
  logging.info("Args of this run are:\n")
  for key, value in vars(myargs).items():
    logging.info(f"Key: {key}, Value: {value}")

# Preparing args, By having args variable here, global, it becomes visible everywhere
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# Set the PYTORCH_CUDA_ALLOC_CONF environment variable # for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.5,max_split_size_mb:512'


# args = prepare_args() # *** uncommnet when you are using terminal to run your code on your own server ***
args = prepare_args(running_on_notebook=True, args=["--checkpoint_address", "facebook/bart-large", "--tokenizer_address", 
                                                    "facebook/bart-large", "--raw_data_address", "./raw_disflQA_filled_missings_of_train_dev",
                                                    "--job_name", "FluentAI_v2.2", "--train", "--evaluate"])


torch.set_printoptions(profile="full")
logging.basicConfig(filename=f"{args.job_name}.log", filemode='a', level=logging.DEBUG, force=True,)
print_args_in_logs(args)

tokenizer = BartTokenizer.from_pretrained(args.tokenizer_address)


## 1.3 Utils

def pickle_thing(thing, name):
  # name is the full path + name of the output file
  with open(f'{name}', 'wb') as f:
    pickle.dump(thing, f)

def unpickle(path):
  # Unpickle the pickled object
  with open(path, 'rb') as f:
    unpickled_thing = pickle.load(f)
  return unpickled_thing

def load_json(name):
    with open(name) as f:
      data = json.load(f)
    return data


def cosine_similarity(vector1, vector2):
  """
  Calculate the cosine similarity between two numpy vectors.
  """
  dot_product = np.dot(vector1, vector2)
  norm_vector1 = np.linalg.norm(vector1)
  norm_vector2 = np.linalg.norm(vector2)
  similarity = dot_product / (norm_vector1 * norm_vector2)
  return similarity.item()

def compute_mean_and_confidence_interval(data, n_iterations=1000, confidence_level=95):
  """
  Calculate the mean and confidence interval of a list of values using the bootstrap resampling method.
  """
  assert 0 < confidence_level < 100, "Confidence level must be between 0 and 100"
  lower_percentile = (100 - confidence_level) / 2
  upper_percentile = 100 - lower_percentile
  replicates_mean = np.zeros(n_iterations) # mean values of bootstrap replicates


  for i in range(n_iterations):
    bootstrap_replicate = np.random.choice(data, size=len(data), replace=True)
    replicates_mean[i] = np.mean(bootstrap_replicate)

  final_mean = np.mean(replicates_mean) # final mean value of the estimate (in our case ROUGE)
  confidence_interval = np.percentile(replicates_mean, [lower_percentile, upper_percentile])
  margin_of_error = (confidence_interval[1] - confidence_interval[0])/ 2

  return final_mean, confidence_interval, margin_of_error

def tokens_and_confidences(outputs):
  """
  Function to extract the generated sequence of the first beam and calculate confidence for each token in the first (best) beam sequence
  """
  # Extract the generated sequence of the first beam
  first_beam_sequence = outputs.sequences[0]

  # Extract the logits/scores from the model for each step of the generation process for the beam id zero
  token_logits = torch.stack(outputs.scores, dim=1)[0].to(device)  # Shape: (num_tokens, vocab_size)

  # Convert logits to probabilities using softmax
  token_probs = F.softmax(token_logits, dim=-1)  # Shape: (num_tokens, vocab_size)

  # Extract the token ids of the first beam
  generated_token_ids = first_beam_sequence[1:]  # Remove the <BOS> token

  # Calculate confidence for each token in the first beam sequence
  token_confidences = token_probs[range(len(generated_token_ids)), generated_token_ids]
  confidences_list = [round(conf.item(),4) for conf in token_confidences]

  assert len(generated_token_ids) == len(confidences_list)

  # Calculating average confidence without start and end tokens
  avg_confidence = sum(confidences_list[1:-1])/len(confidences_list[1:-1])
  avg_confidence = round(avg_confidence,4)

  return generated_token_ids.tolist(), confidences_list, avg_confidence


@torch.no_grad()
def correct_disfluency(disfluent_question, tokenizer, model):
  """
  Function to generate fluent question from disfluent input
  """
  inputs = tokenizer(disfluent_question, return_tensors="pt", max_length=64, truncation=True).to(device)
  outputs = model.generate(inputs["input_ids"], min_length=3, max_length=64, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.1, early_stopping=True,)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

@torch.no_grad()
def generate_result_json_file(data, model, embedding_model):
  """
  Function to generate results json file
  """
  print("Generating results json file ...")
  logging.info("Generating results json file ...")
  n = len(data["raw_test_data"])
  res = {i:{"BLEU_scores": None, "GLEU":None, "semantic_sim_nomicAI":None,
            "avg_confidence":None, "confidences_list":None, "generated_token_ids":None,
            "corrected":None, "disfluent":None, "fluent":None} for i in range(n)}

  for i, (k,v) in enumerate(data["raw_test_data"].items()):
    res[i]["disfluent"] = v["disfluent"]
    res[i]["fluent"] = v["original"]
    inputs = tokenizer(v["disfluent"], return_tensors="pt", max_length=args.max_encoder_input_lenght, truncation=True).to(device)
    # Generate output with return dict in generate and output scores set to True to return logits for computing confidence
    outputs = model.generate(
      inputs["input_ids"],
      min_length=args.min_decoder_output_lenght,
      max_length=args.max_decoder_output_lenght,
      num_beams=args.num_beams,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
      repetition_penalty=args.repetition_penalty,
      early_stopping=True,
      return_dict_in_generate=True,
      output_scores=True)

    res[i]["generated_token_ids"], res[i]["confidences_list"], res[i]["avg_confidence"] = tokens_and_confidences(outputs)
    res[i]["corrected"] = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    res[i]["BLEU_scores"] = sacrebleu.corpus_bleu([res[i]["corrected"]], [[res[i]["fluent"]]]).precisions # Fluent is the reference (or oracle)
    res[i]["GLEU"] = nltk.translate.gleu_score.sentence_gleu([word_tokenize(res[i]["fluent"])], word_tokenize(res[i]["corrected"]))
    res[i]["semantic_sim_nomicAI"] = round(cosine_similarity(embedding_model.encode('search_query: '+ res[i]["fluent"], show_progress_bar=False), embedding_model.encode('search_query: ' + res[i]["corrected"],show_progress_bar=False)), 4)

    # if i == 10:
    #   print("Printing result to make sure it is running fine")
    #   print(json.dumps(res[1], indent=3))
    #   print(json.dumps(res[2], indent=3))
    #   print(json.dumps(res[3], indent=3))
    #   break

  # Stroing res as a json file
  with open(f'{args.job_name}_results.json', 'w') as f:
    json.dump(res, f)
  return res

def print_table_of_results(results):
  """
  Function to print table of results
  """
  pm = "\u00B1"
  b1, b2, b3, b4, gleu, sim, conf = [], [], [], [], [], [], []

  for k, v in results.items():
      b1.append(v["BLEU_scores"][0])
      b2.append(v["BLEU_scores"][1])
      b3.append(v["BLEU_scores"][2])
      b4.append(v["BLEU_scores"][3])
      gleu.append(v["GLEU"])
      sim.append(v["semantic_sim_nomicAI"])
      conf.append(v["avg_confidence"])

  # Calculate mean and confidence interval for each metric
  metrics = [b1, b2, b3, b4, gleu, sim, conf]
  metric_names = ["B1", "B2", "B3", "B4", "GLEU", "Semantic Sim", "Confidence"]

  # Define the width for each column
  column_width = 15

  # Print header
  header = "| " + " | ".join(f"{name:<{column_width}}" for name in metric_names) + " |"
  print(header)
  print("|" + "-" * (len(header) - 2) + "|")  # Separator line

  # Print results
  results_row = []
  for metric in metrics:
      mean, ci, moe = compute_mean_and_confidence_interval(metric)
      results_row.append(f"{mean:.2f} {pm}{moe:.2f}")

  results_line = "| " + " | ".join(f"{result:<{column_width}}" for result in results_row) + " |"
  print(results_line)


## 1.4 Handling data


# Tokenize the input (disfluent) and target (fluent) questions
def tokenize_function(examples):
    model_inputs = tokenizer(examples["disfluent"], max_length=args.max_encoder_input_lenght, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["fluent"], max_length=args.max_decoder_output_lenght, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def crank_dataset():
  data = {"raw_train_data": None, "train_dataset": None, "tokenized_train_dataset": None, "train_sampler": None, "train_loader": None,
          "raw_val_data": None, "val_dataset": None, "tokenized_val_dataset": None, "val_sampler": None, "val_loader": None,
          "raw_test_data": None, "test_dataset": None,"tokenized_test_dataset": None, "test_sampler": None, "test_loader": None,}

  logging.info(f'From Rank: {args.rank}, ==> Preparing data ...')

  if args.dataset_objects_address is None:
    # load raw data
    data["raw_train_data"] = load_json(f'{args.raw_data_address}/Disfl_QA_train.json')
    data["raw_val_data"]  = load_json(f'{args.raw_data_address}/Disfl_QA_dev.json')
    data["raw_test_data"]  = load_json(f'{args.raw_data_address}/Disfl_QA_test.json')
    train_fluent, train_disfluent, dev_fluent, dev_disfluent, test_fluent, test_disfluent = [], [], [], [], [], []
    for _,v in data["raw_train_data"].items():
      train_fluent.append(v['original'])
      train_disfluent.append(v['disfluent'])
    for _,v in data["raw_val_data"] .items():
      dev_fluent.append(v['original'])
      dev_disfluent.append(v['disfluent'])
    for _,v in data["raw_test_data"].items():
      test_fluent.append(v['original'])
      test_disfluent.append(v['disfluent'])
    logging.info(f"len(train_fluent), len(dev_disfluent), len(test_fluent): {len(train_fluent)}, {len(dev_disfluent)}, {len(test_fluent)}.")

    # Example disfluent and fluent questions
    train_data = {'disfluent': train_disfluent, 'fluent': train_fluent,}
    dev_data = {'disfluent': dev_disfluent, 'fluent': dev_fluent,}
    test_data = {'disfluent': test_disfluent, 'fluent': test_fluent,}

    # Convert data to Hugging Face Dataset
    data["train_dataset"] = Dataset.from_dict(train_data)
    data["val_dataset"] = Dataset.from_dict(dev_data)
    data["test_dataset"] = Dataset.from_dict(test_data)

    # Tokenize dataset
    data["tokenized_train_dataset"] = data["train_dataset"].map(tokenize_function, batched=True)
    data["tokenized_val_dataset"] = data["val_dataset"].map(tokenize_function, batched=True)

    # pickle datasets
    folder = f'{args.job_name}_ds_objects' # decoder dataset objects
    os.makedirs(folder, exist_ok=True)
    pickle_thing(data["train_dataset"], f"./{folder}/train_dataset.pkl")
    pickle_thing(data["val_dataset"], f"./{folder}/val_dataset.pkl")
    pickle_thing(data["test_dataset"], f"./{folder}/test_dataset.pkl")
  else:
    # Load dataset objects
    data["train_dataset"] = unpickle(f"{args.dataset_objects_address}/train_dataset.pkl")
    data["val_dataset"]  = unpickle(f"{args.dataset_objects_address}/val_dataset.pkl")
    data["test_dataset"]  = unpickle(f"{args.dataset_objects_address}/test_dataset.pkl")
    # load raw files
    data["raw_train_data"] = load_json(f'{args.raw_data_address}/Disfl_QA_train_filled_missings.json')
    data["raw_val_data"]  = load_json(f'{args.raw_data_address}/Disfl_QA_dev_filled_missings.json')
    data["raw_test_data"]  = load_json(f'{args.raw_data_address}/Disfl_QA_test.json')
  return data


## 1.5 Model
## We don't need to redefine a model. We will use BART model and finetune it at this stage


## 1.6 Train/validate/test functions

## Define training/fine-tuning arguments
training_args = TrainingArguments(
    output_dir=f"./{args.job_name}_outputs_dir",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.max_epochs,
    weight_decay=0.01,
    logging_dir=f"./{args.job_name}_logs",
    save_total_limit=10,
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="eval_loss",  # Specify which metric to use for selecting the best model
    greater_is_better=False,  # For loss, lower values are better
    warmup_steps=100,
    logging_steps=100,
    fp16=False,  # Note: don't use it. it will make training unstable
)

## 1.7 Evaluation functions



## 1.8 Driver function
def main():
  logging.info("Starting...")
  torch.cuda.set_device(args.local_rank) # Set to local_rank not (global) rank

  if args.distributed:
    """ this block initializes a process group and initiate communications
    between all processes running on all nodes """
    logging.info(f'From Rank: {args.rank}, ==> Initializing Process Group...')
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank) #init the process group

  ## Cranking dataset
  logging.info('Cranking the data loaders...')
  data = crank_dataset()


  logging.info(f'From Rank: {args.rank}, ==> Making model, optimizer, and criterion ...')
  model_name = args.checkpoint_address
  model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

  if args.train:
    torch.cuda.empty_cache()
    gc.collect()
    # Initialize Trainer
    trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=data['tokenized_train_dataset'],
          eval_dataset=data['tokenized_val_dataset'],
      )
    # Train the model
    trainer.train()
    # Save the model
    model.save_pretrained(f"./{args.job_name}_best_model")
    tokenizer.save_pretrained(f"./{args.job_name}_best_model")
    logging.info(f'From Rank: {args.rank}, ==> Model saved.')
    torch.cuda.empty_cache()
    gc.collect()

  if args.evaluate:
    # Load embedding model
    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    # Generate results
    results = generate_result_json_file(data, model, embedding_model)
    # Print table of results
    print_table_of_results(results)


if __name__ == "__main__":
  if args.debug:
    logging.info(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode('utf-8'))

  # driver function
  main()

  logging.info("... Finished")