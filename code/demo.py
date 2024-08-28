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
  running_on_colab = False 
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_address', type=str, default='FluentAI_v2.2_best_model' )
  parser.add_argument('--disfluent_input', type=str, default="Disfluent" )

  if running_on_colab:
    # if running the code on google colab
    args = parser.parse_args(args=[])
    # example Where is the closest coffeeshop sorry gym?
    args.disfluent_input = "Your Disfluent Question Here" 
    args.disfluent_input = "Where is the closest coffeeshop sorry gym?" 
  else:
    args = parser.parse_args()

  tokenizer = BartTokenizer.from_pretrained(args.checkpoint_address)
  model = BartForConditionalGeneration.from_pretrained(args.checkpoint_address).to(device)
  print(correct_disfluency(args.disfluent_input, tokenizer, model))