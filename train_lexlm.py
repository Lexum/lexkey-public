import numpy as np
import torch
import neptune.new as neptune
from lexlm_dataset import DenoisingCollator, LexLMDataset
from transformers import MBartConfig, MBart50TokenizerFast, MBartForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import os
from train_bart import S3Uploader
from lexkey_utils import freeze_mbart_layers


def train_denoising(data_dir, model_name, do_sent_sub=True, checkpoint_path=None, freeze=True):
    model_checkpoint = model_name if checkpoint_path is None else checkpoint_path

    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        evaluation_strategy='steps',
        eval_steps=10000,
        lr_scheduler_type='linear',
        learning_rate=1e-4,
        save_strategy='steps',
        save_steps=10000,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        output_dir=data_dir + "models/" + model_name + "/",
        generation_max_length=1024,
        report_to="neptune")

    if checkpoint_path is not None:
        training_args.resume_from_checkpoint = checkpoint_path

    sample_frac = 1.0
    multilingual = True

    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    train_data = LexLMDataset(data_dir + "global_train.csv",
                              tokenizer, data_dir, multilingual, sample=sample_frac, split_doc=False)
    valid_data = LexLMDataset(data_dir + "global_valid.csv",
                              tokenizer, data_dir, multilingual, sample=sample_frac, split_doc=False)

    data_collator = DenoisingCollator(tokenizer, do_sent_sub)
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

    if freeze:
        freeze_mbart_layers(model)

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        data_collator=data_collator)

    trainer.train()
    print(trainer.evaluate())


if __name__ == "__main__":
    data_dir = "~/lexkey/data/"
    tokenizer_path = None
    s3_bucket = "lexkey-artifacts"

    neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
    neptune_project = os.getenv("NEPTUNE_PROJECT")
    tags = ["lexlm"]
    run = neptune.init(project=neptune_project, api_token=neptune_api_token, tags=tags)
    run_id = run.get_run_url().rsplit('/', 1)[1]
    print("NEPTUNE RUN ID = " + run_id)
    os.environ['NEPTUNE_RUN_ID'] = run_id

    # Upload artifacts to s3
    s3_client = S3Uploader(s3_bucket)

    train_path = data_dir + "global_train.csv"
    valid_path = data_dir + "global_valid.csv"

    # Stop run so report_to can resume it.
    run.stop()

    np.random.seed(42)
    train_denoising(data_dir, "facebook/mbart-large-50", do_sent_sub=True, checkpoint_path=None)

    # Uploading created files afiter run end.
    run = neptune.init(project=neptune_project, api_token=neptune_api_token, run=run_id)
