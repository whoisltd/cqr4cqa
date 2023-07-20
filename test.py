import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from src.modules.seq2seq_trainer import GPT2Seq2SeqTrainer

from src.data_utils.canard import load_canard
from src.utils.utils import build_compute_metrics_fn_gpt2, add_special_tokens_, postprocess_gpt2_predictions

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
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "Path to pre-trained model or shortcut name selected"
        },
    )
    model_type: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Model type selected"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="canard", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    history_len: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of history utterances will be concatenated into the inputs."
        },
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Length of the input sequences"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # add_special_tokens: Optional[bool] = field(
    #     default=True, metadata={"help": "Whether to add special tokens in the training process"}
    # )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    batchify: bool = field(
        default=False, metadata={"help": "Prepare the dataset in batch mode."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class ExtraArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls."}
    )        
    from_scratch: Optional[bool] = field(
        default=False,
        metadata={"help": "Train the model from scratch without the pretrained weights."}
    ) 
os.environ["WANDB_DISABLED"] = "true"

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, ExtraArguments))

model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

config = AutoConfig.from_pretrained(model_args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

lm_datasets = load_canard(data_args, tokenizer, data_dir='src/data/canard', output_dir='src/data/canard', overwrite_cache=False, model_type="seq2seq")

model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

add_special_tokens_(model, tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
# save datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_type = "decoder_only"


early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=extra_args.early_stopping_patience)

trainer = GPT2Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"] if training_args.do_train else None,
            eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator, 
            # compute_metrics=compute_metrics_fn,
            # post_process_function=post_processing_function,
            # eval_from_path=data_args.dataset_name == "canard",
            callbacks=[early_stopping_callback],
        )

if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(resume_from_checkpoint=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    eval_output["perplexity"] = perplexity

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(eval_output.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
import pickle
#save train dataset
with open('src/data/canard/train_dataset.pkl', 'wb') as f:
    pickle.dump(lm_datasets['train'], f)
#save validation dataset
with open('src/data/canard/valid_dataset.pkl', 'wb') as f:
    pickle.dump(lm_datasets['validation'], f)
#save test dataset
with open('src/data/canard/test_dataset.pkl', 'wb') as f:
    pickle.dump(lm_datasets['test'], f)
