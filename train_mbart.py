import pandas as pd
import datasets
from torch.utils.data import Dataset
#from datasets import Dataset
from datasets import load_metric
from transformers import (
    HfArgumentParser,
    SchedulerType,
    set_seed,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
    MBartForConditionalGeneration,
    BartTokenizerFast,
    MBart50TokenizerFast,
    LEDForConditionalGeneration,
    LEDTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer)
import nltk
import numpy as np

import time
import datetime
import os
import csv

import boto3
from botocore.exceptions import ClientError
import hashlib
import logging
import shutil

import neptune.new as neptune
from bert_score import BERTScorer

from dataclasses import dataclass, field
from typing import Optional

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers.deepspeed import is_deepspeed_zero3_enabled

from lexkey_utils import freeze_mbart_layers, get_stopping_criterias

nltk.download('punkt')


@dataclass
class TrainingArguments:
    model_name: str = field(metadata={"help": "Model identifier from huggingface.co/models"})
    project_name: str = field(metadata={"help": "Project name for saving"})
    data_dir: str = field(metadata={"help": "Data Directory."})
    train_file: str = field(metadata={"help": "The input training data file (a jsonlines or csv file)."})
    validation_file: str = field(
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        }
    )
    test_file: str = field(
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        }
    )
    sample_file: str = field(
        default=None,
        metadata={
            "help": "An optional input sample data file to generate predictions."
        }
    )
    model_path: str = field(
        default=False,
        metadata={"help": "Path to model."}
    )
    tokenizer_path: str = field(
        default=False,
        metadata={"help": "Path to tokenizer."}
    )
    load_from_disk: bool = field(
        default=False,
        metadata={"help": "Whether to load model from disk."}
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_num_beams: int = field(
        default=1,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                    "to the `num_beams` value of the model configuration."
        }
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_generate_sample: bool = field(default=False, metadata={"help": "Whether to generate predictions on a sample set."})
    predict_datasets: Optional[str] = field(
        default="test",
        metadata={"help": "Datasets to use in predict step, comma-separated values"})
    bert_score: bool = field(default=False, metadata={"help": "Whether to run BertScore metrics on the test set."})
    language: Optional[str] = field(default=None, metadata={"help": "Language used in training and datasets"})
    lr: Optional[str] = field(default=2e-5, metadata={"help": "learning rate, string will be converted to float"})
    bs: Optional[int] = field(default=16, metadata={"help": "Batch Size"})
    eval_bs: Optional[int] = field(default=16, metadata={"help": "Batch Size"})
    epochs: Optional[int] = field(default=5, metadata={"help": "Number of epochs"})
    scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    text_column: str = field(
        default="texts",
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default="keywords",
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    output_dir: Optional[str] = field(default='.', metadata={"help": "Output Directory."})
    max_input_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        }
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        }
    )
    max_predict_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        }
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    tags: Optional[str] = field(
        default=None,
        metadata={"help": "Neptune tags, comma-separated values"})
    s3_bucket: Optional[str] = field(
        default="lexkey-artifacts",
        metadata={"help": "S3 bucket for artifacts upload."})
    upload_finetuned_model: bool = field(default=False, metadata={"help": "Whether to upload the fine-tuned model to S3."})
    upload_pretrained_model: bool = field(default=False, metadata={"help": "Whether to upload the pre-trained model to S3."})
    upload_tokenizer: bool = field(default=False, metadata={"help": "Whether to upload the tokenizer to S3."})
    led: bool = field(default=False, metadata={"help": "Whether it's a LED model."})
    lsg: bool = field(default=False, metadata={"help": "Whether it's a LSG model."})
    freeze: bool = field(default=False, metadata={"help": "Whether to freeze layers in a mbart model."})
    min_prob: Optional[float] = field(default=None, metadata={"help": "The minimum word probability before stopping generation"})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 instead of fp16."})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={"help": "The maximum size of repeated n_grams of tokens during generation"})


class LexSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_length": self.args.generation_max_length if self.args.generation_max_length is not None else self.model.config.max_length,
            "num_beams": self.args.generation_num_beams if self.args.generation_num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if args.min_prob is not None:
            gen_kwargs["stopping_criteria"] = get_stopping_criterias(args.max_target_length, args.min_prob)
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        if self.tokenizer is not None:
            generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
            # very ugly hack to make it work
            generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
        else:
            generation_inputs = {"input_ids": inputs["input_ids"]}

        labels_length_tokens = inputs["labels"][:, 1:2] if has_labels else None

        bs = inputs["labels"].shape[0]
        decoder_input_ids = torch.tensor([2, 0], device=self.args.device) # generations always start with these 2 tokens, <pad> and </n>
        decoder_input_ids = decoder_input_ids.repeat(bs, 1)
        decoder_input_ids = torch.cat((decoder_input_ids, labels_length_tokens), 1) # concat length tokens

        model_kwargs = {
            "decoder_input_ids": decoder_input_ids
        }

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
            **model_kwargs
        )

        if args.min_prob is not None:
            # This returns a dict.
            generated_tokens = generated_tokens.sequences

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)


class LexKeyDataset(Dataset):
    def __init__(self, data_path, tokenizer, mbart, data_dir, language='en', led=False, max_input_length=1024, max_target_length=1024, sample=False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_dir = data_dir + "texts"
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.language = language
        self.sample = sample
        self.led = led
        self.mbart = mbart

        if data_path is not None:
            self.dataframe = pd.read_csv(data_path)
            if language is not None:
                self.dataframe = self.dataframe[self.dataframe.language == self.language]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = self.get_text(row.filename.replace(".html", ".txt"))
        toks = self.tokenize(text, row.language, self.max_input_length)
        if not self.sample:
            labels = self.tokenize(row.keywords, row.language, self.max_target_length)
            toks["labels"] = labels["input_ids"]

        if self.led:
            global_attention_mask = [0] * len(toks["input_ids"])
            global_attention_mask[0] = 1
            global_attention_mask[self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)] = 1 # global attention on eos token
            toks["global_attention_mask"] = global_attention_mask

        return toks

    def get_text(self, filename):
        with open(self.data_dir + filename, "r", encoding='utf-8') as f:
            return f.read()

    def tokenize(self, text, lang, max_length):
        if self.mbart:
            lang_tag = lang + "_XX"
            self.tokenizer.src_lang = lang_tag
            self.tokenizer.tgt_lang = lang_tag

        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_token_type_ids=False)
        return tokens

    def get_raw_item(self, idx):
        return self.dataframe.iloc[idx]


class LexKeyDataSets:
    def __init__(self, tokenizer, mbart, train_path, valid_path, test_path, sample_path, data_dir, lang, led, max_input_length, max_target_length):
        self.tokenizer = tokenizer

        self.data_sets_labels = ["train", "valid", "test", "sample"]
        self.paths = {
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
            "sample": sample_path
        }

        self.lang = lang
        self.data_dir = data_dir
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.mbart = mbart
        self.led = led
        self.data_dicts = {}
        self.tokenized_data_dicts = {}

    def prepare_data(self):
        for label in self.data_sets_labels:
            self.prepare_data_set(label)

    def prepare_data_set(self, data_set_label):
        df = LexKeyDataset(self.paths[data_set_label],
                           self.tokenizer,
                           self.mbart,
                           self.data_dir,
                           self.lang,
                           self.led,
                           self.max_input_length,
                           self.max_target_length,
                           data_set_label == "sample")

        print(f'Shape of {data_set_label} : {df.dataframe.shape}')

        if self.lang is None:
            subset_en = LexKeyDataset(self.paths[data_set_label],
                                      self.tokenizer,
                                      self.mbart,
                                      self.data_dir,
                                      "en",
                                      self.led,
                                      self.max_input_length,
                                      self.max_target_length,
                                      data_set_label == "sample")
            subset_fr = LexKeyDataset(self.paths[data_set_label],
                                      self.tokenizer,
                                      self.mbart,
                                      self.data_dir,
                                      "fr",
                                      self.led,
                                      self.max_input_length,
                                      self.max_target_length,
                                      data_set_label == "sample")

            dd = datasets.DatasetDict({"all": df, "en": subset_en, "fr": subset_fr})
        else:
            dd = datasets.DatasetDict({"all": df})

        self.data_dicts[data_set_label] = dd

    def get_data(self, label):
        return self.data_dicts[label]


def main():
    start_point = datetime.datetime.now()

    print('Start time:', start_point.strftime("%Y-%m-%d %H:%M:%S"))
    # Set seed before initializing model.
    set_seed(args.seed)

    print('Loading data...')
    if args.led:
        if args.model_name == "facebook/mbart-large-50":
            tokenizer = MBart50TokenizerFast.from_pretrained(
                args.model_name if not args.tokenizer_path else args.tokenizer_path, src_lang="en_XX", tgt_lang="en_XX", model_max_length=args.max_input_length)
        else:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                args.model_name if not args.tokenizer_path else args.tokenizer_path, model_max_length=args.max_input_length)
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    elif args.lsg:
        if args.tokenizer_path:
            tok_path = args.tokenizer_path
        else:
            tok_path = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    else:
        if args.model_name == "facebook/mbart-large-50":
            tokenizer = MBart50TokenizerFast.from_pretrained(
                args.model_name if not args.tokenizer_path else args.tokenizer_path, src_lang="en_XX", tgt_lang="en_XX")
        else:
            tokenizer = BartTokenizerFast.from_pretrained(
                args.model_name if not args.tokenizer_path else args.tokenizer_path)

    print('mbart? ', args.model_name == "facebook/mbart-large-50")

    canlii_datasets = LexKeyDataSets(
        tokenizer,
        args.model_name == "facebook/mbart-large-50",
        args.data_dir + args.train_file,
        args.data_dir + args.validation_file,
        args.data_dir + args.test_file,
        args.data_dir + args.sample_file if args.sample_file else None,
        args.data_dir,
        args.language,
        args.led,
        args.max_input_length,
        args.max_target_length,
    )

    # Loading data
    start = time.perf_counter()
    if args.do_train:
        canlii_datasets.prepare_data_set("train")
        canlii_datasets.prepare_data_set("valid")

    if args.do_eval and not args.do_train:
        canlii_datasets.prepare_data_set("valid")

    if args.do_predict:
        canlii_datasets.prepare_data_set("test")

    if args.do_generate_sample:
        canlii_datasets.prepare_data_set("sample")

    time_taken = time.perf_counter() - start
    print(f'Loading time: {datetime.timedelta(seconds=time_taken)}')

    # Neptune configuration
    if args.do_train:
        neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
        neptune_project = os.getenv("NEPTUNE_PROJECT")
        tags = args.tags.split(",") if args.tags else None
        run = neptune.init(project=neptune_project, api_token=neptune_api_token, tags=tags)
        run_id = run.get_run_url().rsplit('/', 1)[1]
        print("NEPTUNE RUN ID = " + run_id)
        os.environ['NEPTUNE_RUN_ID'] = run_id

    print(f"Training/evaluation parameters {args}")


    # Preparing training
    print('Preparing training...')
    start = time.perf_counter()

    if args.led:
        model = LEDForConditionalGeneration.from_pretrained(
            args.model_path if args.load_from_disk else args.model_name)
    elif args.lsg:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        if args.model_name == "facebook/mbart-large-50":
            model = MBartForConditionalGeneration.from_pretrained(
                args.model_path if args.load_from_disk else args.model_name)
            if args.freeze:
                freeze_mbart_layers(model)
        else:
            model = BartForConditionalGeneration.from_pretrained(
                args.model_path if args.load_from_disk else args.model_name)

    if args.do_train:
        run["parameters/max_input_length"] = args.max_input_length

    #Stop run to resume with HF -> Workaround
    if args.do_train:
        run.stop()

    training_args = Seq2SeqTrainingArguments(
        args.project_name,
        report_to="neptune",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.lr),
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=args.weight_decay,
        save_total_limit=5,
        num_train_epochs=args.epochs,
        predict_with_generate=args.predict_with_generate,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.max_target_length,
        overwrite_output_dir=True,
        fp16=(not args.bf16),
        bf16=args.bf16,
        lr_scheduler_type=SchedulerType(args.scheduler_type),
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=16
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    trainer = LexSeq2SeqTrainer(
        model,
        training_args,
        train_dataset=canlii_datasets.get_data("train")["all"] if args.do_train else None,
        eval_dataset=canlii_datasets.get_data("valid")["all"] if args.do_eval or args.do_train else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.do_train:
        time_taken = time.perf_counter() - start
        print(f'Preparing training time: {datetime.timedelta(seconds=time_taken)}')
        print()

        # Training
        print('##### Training #####')
        start = time.perf_counter()

        train_result = trainer.train()

        time_taken = time.perf_counter() - start
        print(f'Training time: {datetime.timedelta(seconds=time_taken)}')

        metrics = train_result.metrics

        print("Saving model and metrics")
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        print('##### Evaluation / Validation Data #####')
        if args.language is None:
            datasets_to_evaluate = ["en", "fr"]
        else:
            datasets_to_evaluate = ["all"]

        for to_evaluate in datasets_to_evaluate:
            eval_dataset = canlii_datasets.get_data("valid")[to_evaluate]
            metrics = trainer.evaluate(eval_dataset=eval_dataset, max_length=args.max_target_length,
                                       metric_key_prefix="eval_valid_" + to_evaluate)
            metrics["eval_samples"] = min(args.max_eval_samples, len(eval_dataset))

            print(f'Evaluation metrics for validation data / {to_evaluate} : {metrics}')
            trainer.log_metrics("Evaluation : validation data / " + to_evaluate, metrics)
            trainer.save_metrics("Evaluation_validation_data_" + to_evaluate, metrics)

        print('##### Evaluation / Train Data #####')
        for to_evaluate in datasets_to_evaluate:
            eval_dataset = canlii_datasets.get_data("train")[to_evaluate]
            metrics = trainer.evaluate(eval_dataset=eval_dataset, max_length=args.max_target_length,
                                       metric_key_prefix="eval_train_" + to_evaluate)
            metrics["eval_samples"] = min(args.max_eval_samples, len(eval_dataset))

            print(f'Evaluation metrics for train data / {to_evaluate} : {metrics}')
            trainer.log_metrics("Evaluation : train data / " + to_evaluate, metrics)
            trainer.save_metrics("Evaluation_train_data_" + to_evaluate, metrics)

    # Prediction
    if args.do_predict:
        print('##### Predict #####')
        datasets_to_predict = args.predict_datasets.split(",")
        for to_predict in datasets_to_predict:
            print('#### Predict : ' + to_predict + ' #####')
            predict_dataset = canlii_datasets.get_data(to_predict)["all"]
            if args.min_prob is not None:
                predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict_" + to_predict,
                                                  stopping_criteria=get_stopping_criterias(args.max_target_length,
                                                                                           args.min_prob))
            else:
                predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict_" + to_predict,
                                                  max_length=args.max_target_length)
            metrics = predict_results.metrics
            metrics["predict_samples"] = min(args.max_predict_samples, len(predict_dataset))

            trainer.log_metrics("prediction", metrics)
            trainer.save_metrics("prediction", metrics)

            to_save = []
            fieldnames = ["cc2DecisionId", "language", "path", "decisionTribunal",
                          "label", "prediction", "rouge1", "rouge2", "rougeL", "rougeLsum"]

            if trainer.is_world_process_zero():
                if args.predict_with_generate:
                    predictions = predict_results.predictions
                    labels = predict_results.label_ids

                    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=True)

                    print('##### Metrics #####')
                    if args.bert_score:
                        fieldnames.extend(["BertScore/Precision", "BertScore/Recall", "BertScore/F1"])
                        bert_scorer_en = BERTScorer(lang="en", rescale_with_baseline=True)
                        bert_scorer_fr = BERTScorer(lang="fr", rescale_with_baseline=True)

                    index = 0
                    for pred, label in zip(decoded_preds, decoded_labels):
                        row = predict_dataset.get_raw_item(index)
                        language = row["language"]
                        # rouge metrics
                        result = metric.compute(predictions=[pred], references=[label], use_stemmer=True)
                        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                        result["label"] = label
                        result["prediction"] = pred

                        # Bert score
                        if args.bert_score:
                            bert_scorer = bert_scorer_fr if language == "fr" else bert_scorer_en
                            precision, recall, f1 = bert_scorer.score([pred], [label])
                            result["BertScore/Precision"] = precision.item() * 100
                            result["BertScore/Recall"] = recall.item() * 100
                            result["BertScore/F1"] = f1.item() * 100

                        # add additional metadata for analysis
                        result["language"] = language
                        result["cc2DecisionId"] = row["cc2DecisionId"]
                        result["path"] = row["path"]
                        result["decisionTribunal"] = row["decisionTribunal"]
                        index = index + 1

                        to_save.append(result)

                    output_prediction_file = os.path.join(args.output_dir,
                                                          args.project_name + "_" + to_predict + "_predictions_metrics.csv")
                    with open(output_prediction_file, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(to_save)

    if args.do_generate_sample:
        print('##### Generation for sample #####')
        predict_dataset = canlii_datasets.get_data("sample")["all"]
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict_sample",
                                      max_length=args.max_target_length)

        to_save = []
        fieldnames = ["cc2DecisionId", "language", "title", "citation", "path", "decisionTribunal", "prediction"]

        if trainer.is_world_process_zero():
            if args.predict_with_generate:
                predictions = predict_results.predictions

                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)

                index = 0
                for pred in decoded_preds:
                    row = predict_dataset.get_raw_item(index)
                    result = {"cc2DecisionId": row["cc2DecisionId"],
                              "language": row["language"],
                              "title": row["title"],
                              "citation": row["citation"],
                              "path": row["path"],
                              "decisionTribunal": row["decisionTribunal"],
                              "prediction": pred}
                    to_save.append(result)
                    index = index + 1

                output_prediction_file = os.path.join(args.output_dir,
                                                      args.project_name + "_sample_predictions.csv")
                with open(output_prediction_file, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(to_save)


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(f"Training/evaluation parameters {args}")
    data_dir = args.data_dir
    main()
