import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from src.modules.seq2seq_trainer import GPT2Seq2SeqTrainer
from src.data_utils.canard import load_canard
import argparse

from src.utils.utils import add_special_tokens_

#arg values
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="canard", type=str)
parser.add_argument("--dataset_config_name", default=None, type=str)
parser.add_argument("--train_file", default=None, type=str)
parser.add_argument("--validation_file", default=None, type=str)
parser.add_argument("--history_len", default=3, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--overwrite_cache", default=False, type=bool)
parser.add_argument("--validation_split_percentage", default=5, type=int)


from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
    set_seed,
    TrainingArguments,
    Trainer,
)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)



config = AutoConfig.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')


lm_datasets = load_canard(parser.parse_args(), tokenizer, data_dir='src/data/canard', output_dir='src/data/canard', overwrite_cache=False, model_type="seq2seq")

model = AutoModelForCausalLM.from_pretrained(
    'gpt2',
    from_tf=bool(".ckpt" in 'gpt2'),
    config=config,
    revision='main',
    use_auth_token=None,
    )

add_special_tokens_(model, tokenizer)
model.config.pad_token_id = tokenizer.eos_token_id

model_type = "decoder_only"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    )

trainer.train()