import numpy as np

from functools import lru_cache
import torch
from torch import nn
from torch.nn import Module

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer as HFTrainer, TrainingArguments

class CodeSearch(Module):
  def __init__(
    self,
    name = "EleutherAI/gpt-neo-125M",
    cache_dir = "./raw_stack/"
  ):
    super().__init__()
    # create tokenizer and add special token
    self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = cache_dir)
    self.tokenizer.add_special_tokens({'pad_token': '<|endofseq|>'})
    
    # create the code encoder and doc encoder models
    self.code_encoder = AutoModelForCausalLM.from_pretrained(name, cache_dir = cache_dir).transformer
    self.doc_encoder = AutoModelForCausalLM.from_pretrained(name, cache_dir = cache_dir).transformer
    
    self.ln_code = torch.nn.LayerNorm(self.code_encoder.embed_dim)
    self.ln_doc = torch.nn.LayerNorm(self.doc_encoder.embed_dim)
    
    # expant
    self.code_encoder.resize_token_embeddings(self.tokenizer.vocab_size + 1)
    self.doc_encoder.resize_token_embeddings(self.tokenizer.vocab_size + 1)

    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

  def prepare(self, strings, n = 512):
    bsize = len(strings)
    data = self.tokenizer(strings, return_tensors = "pt", max_length = n, padding = "max_length", truncation = True)
    data["input_ids"][range(bsize), -1] = (torch.ones(bsize)*self.tokenizer.vocab_size).long()
    return data

  @lru_cache()
  def _get_arange(self, n):
    return torch.arange(n).to(self.code_encoder.device)

  def forward(
    self,
    code_ids,
    code_ids_attention_mask,
    doc_ids,
    doc_ids_attention_mask,
  ):
    code_out = self.code_encoder(input_ids = code_ids, attention_mask = code_ids_attention_mask)
    doc_out = self.doc_encoder(input_ids = doc_ids, attention_mask = doc_ids_attention_mask)
    
    code_emb = code_out.last_hidden_state[range(len(code_out.last_hidden_state)), code_ids.argmax(1)]
    doc_emb = doc_out.last_hidden_state[range(len(doc_out.last_hidden_state)), doc_ids.argmax(1)]
    
    code_features = self.ln_code(code_emb)
    doc_features = self.ln_doc(doc_emb)
    
    code_features = code_features / code_features.norm(dim=1, keepdim=True)
    doc_features = doc_features / doc_features.norm(dim=1, keepdim=True)

    # CLIP objective 
    logit_scale = self.logit_scale.exp()
    logits_per_code = logit_scale * code_features @ doc_features.t()
    logits_per_doc = logits_per_code.t()
    loss = nn.CrossEntropyLoss()(logits_per_code, self._get_arange(len(code_features)))
    loss += nn.CrossEntropyLoss()(logits_per_doc, self._get_arange(len(code_features)))

    # accuracy
    accuracy = (logits_per_code.argmax(1) == self._get_arange(len(code_features))).float().mean()

    return {
      "loss": loss,
      "accuracy": accuracy,
    }


class Trainer(HFTrainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    print(inputs)
    exit()
    outputs = model(
      code_ids = inputs["code_ids"],
      code_ids_attention_mask = inputs["code_ids_attention_mask"],
      doc_ids = inputs["doc_ids"],
      doc_ids_attention_mask = inputs["doc_ids_attention_mask"],
    )
    loss = outputs["loss"]
    return (loss, outputs) if return_outputs else loss


class Compiler():
  def __init__(
    self,
    cache_dir: str = "./code_search/model/",
    train_ds: str = "./code_search/data/",
    output_dir: str = "./code_search/outputs/",
  ):
    self.model = CodeSearch(cache_dir = cache_dir)
    pytorch_total_params = sum(p.numel() for p in self.model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")

    # # check if cuda is available and then move the model to CUDA
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.model.to(self.device)

    # pass everything to the huggingface trainer
    args = TrainingArguments(
      output_dir = output_dir,
      do_train = True,
      gradient_accumulation_steps = 1,
      learning_rate = 1e-4,
      weight_decay = 0.01,
      max_steps = 1000,
    )

    print(args.to_json_string())

    ds = load_dataset(train_ds, "text")["train"]["data"]

    self.trainer = Trainer(
      model = self.model,
      args = args,
      train_dataset = ds
    )

  def train(self):
    
    # code_out = self.model.prepare([x["code"] for x in ds], 512)
    # doc_out = self.model.prepare([x["docstring"] for x in ds], 128)

    # data = {
    #   "code_ids": code_out["input_ids"],
    #   "code_ids_attention_mask": code_out["attention_mask"],
    #   "doc_ids": doc_out["input_ids"],
    #   "doc_ids_attention_mask": doc_out["attention_mask"],
    # }
    # data = {k:v.to(self.device) for k,v in data.items()}
    # print({k:(v.size(), v.device) for k,v in data.items()})

    # out = self.model(**data)
    # print(out)

    self.trainer.train()


if __name__ == "__main__":
  # compiler = Compiler()
  # compiler.train()

  ds = load_dataset("./code_search/data", "text")["train"]["data"]
  print(ds)

