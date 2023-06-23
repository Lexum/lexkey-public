import nltk
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration, MBartForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq
import difflib
from tqdm import tqdm
from pprint import pprint

tqdm.pandas()
delimiters = ["'", '"', '«', '»', '(', ')', '[', ']', '{', '}', '<', '>']

def cap_sentence(input_text):
    words = nltk.word_tokenize(input_text)
    tagged_words = nltk.pos_tag([word.lower() for word in words])
    capitalized_words = [w.capitalize() if t in ["NN", "NNS"] else w for (w, t) in tagged_words]
    capitalized_words[0] = capitalized_words[0].capitalize()
    return " ".join(capitalized_words)


def nltk_baseline(input_text):
    
    top_level = re.split(r" [|] ", input_text) 
    truecased_text = []
    for t in top_level:
        keywords = re.split(r" — ", t)
        keyword_texts = []
        for k in keywords:
            sentences = nltk.sent_tokenize(k, language='english')    
            sentences_capitalized = [cap_sentence(s) for s in sentences]
            keyword_texts.append(re.sub(" (?=[\.,'!?:;])", "", " ".join(sentences_capitalized)))
        truecased_text.append(" — ".join(keyword_texts))
    
    return " | ".join(truecased_text)


class TruecasingDataset:
    def __init__(self, tokenizer, data_path, text_dir, annotate=False, bilingual=False):
        self.annotate = annotate
        self.bilingual = bilingual
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_path)
        if not bilingual:
            self.data = self.data[self.data.language == 'en']
        self.data = self.data[self.data.decisionTribunal == 'csc-scc']
        self.data["keywords_lc"] = self.data["keywords"].str.lower()
        self.text_dir = text_dir
        
    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_annotation_target(keyword):
        if keyword.isupper():
            return '+'
        elif (keyword[0] in delimiters) and (len(keyword) > 1):
            return keyword[1]
        else:
            return keyword[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row.keywords_lc
        inputs = self.tokenize(text, row.language)
        if self.annotate:
            keywords = re.split("[ -]", row.keywords)
            targets = [self.get_annotation_target(k) for k in keywords if len(k) > 0]
            inputs["labels"] = self.tokenize(" ".join(targets), row.language).input_ids
        else:
            inputs["labels"] = self.tokenize(row.keywords, row.language).input_ids
        
        return inputs
        
    def get_text(self, filename):
        with open(self.text_dir + filename, "r", encoding='utf-8') as f:
            return f.read()
        
    def tokenize(self, text, lang):
        if self.bilingual:
            lang_tag = lang + "_XX"
            self.tokenizer.src_lang = lang_tag
            self.tokenizer.tgt_lang = lang_tag
        return self.tokenizer(text, truncation=True, padding='max_length', max_length=768)


def transformers_baseline():
    # BartForConditionalGeneration
    data_dir = "~/lexkey/data/"
    model_name = "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataset comparing lowercase as input to proper case as label
    train_data = TruecasingDataset(tokenizer, data_dir + "classification_train.csv", data_dir + "texts")
    valid_data = TruecasingDataset(tokenizer, data_dir + "classification_valid.csv", data_dir + "texts")
    
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=20,
        evaluation_strategy='epoch',
        lr_scheduler_type='linear',
        learning_rate=1e-4,
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        output_dir=data_dir + "models/truecasing",
        generation_max_length=1024
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        data_collator=data_collator)
    trainer.train()
    trainer.evaluate()
    
    model.save_pretrained(data_dir + "models/truecasing/best")


def eval_transformers_baseline():
    
    data_dir = "/home/ceratb/lexkey/data/"
    model_name = "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(data_dir + "models/truecasing/best")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_kws = "Constitutional law — Charter of Rights — Fundamental justice — Right to silence — Self‑incrimination " \
               "— Right to fair hearing — Right to make full answer and defence — Evidence — Sexual offences " \
               "— Criminal Code provisions setting out record screening regime to determine admissibility of records" \
               " relating to complainant that are in possession or control of accused — Whether record screening" \
               " regime infringes accused's Charter-protected rights — If so, whether infringement justified " \
               "— Canadian Charter of Rights and Freedoms, ss. 1, 7, 11(c), 11(d) " \
               "— Criminal Code, R.S.C. 1985, c. C‑46, ss. 276, 278.1, 278.92 to 278.94."
    test_kws_lower = test_kws.lower()
    test_sents_lower = test_kws_lower.split(" — ")

    inputs = tokenizer(test_sents_lower, padding=True, return_tensors="pt")
    output = model.generate(inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=1024,
                            num_beams=10, top_k=1, length_penalty=0.0)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    decoded_kws = " — ".join(decoded)
    pprint(list(difflib.context_diff(decoded_kws, test_kws)))
    

def transformers_annotator():
    # BartForConditionalGeneration
    data_dir = "~/lexkey/data/"
    model_name = "facebook/bart-base"
    model_path = data_dir + "models/bart-base-finetuned"
    bilingual = "mbart" in model_name
    annotate = True

    if bilingual:
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    else:
        model = BartForConditionalGeneration.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataset comparing lowercase as input to proper case as label
    train_data = TruecasingDataset(tokenizer,
                                   data_dir + "classification_train.csv",
                                   data_dir + "texts",
                                   annotate=annotate,
                                   bilingual=bilingual)
    valid_data = TruecasingDataset(tokenizer,
                                   data_dir + "classification_valid.csv",
                                   data_dir + "texts",
                                   annotate=annotate,
                                   bilingual=bilingual)
    
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=10,
        evaluation_strategy='epoch',
        lr_scheduler_type='linear',
        learning_rate=1e-4,
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        output_dir=data_dir + "models/truecasing")
    
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args)

    trainer.train()
    trainer.evaluate()
    
    model.save_pretrained(data_dir + "models/truecasing/en_annotator")


def capitalize(kw, cap):
    if (len(kw) == 0) or (len(cap) == 0):
        return kw

    if cap[0].isupper():

        if "." in kw[:-1]:
            return kw.upper()
        elif (kw[0] in delimiters) and (len(kw) > 1):
            return kw[0] + kw[1:].capitalize()
        else:
            return kw.capitalize()
    elif cap[0] == '+':
        return kw.upper()

    return kw


def annotate_one_doc(model, tokenizer, kws, lang_tag=None):
    inputs = tokenizer(kws, padding='max_length', max_length=768, return_tensors="pt")
    inputs.to("cuda")

    if lang_tag:
        output = model.generate(inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                max_length=256,
                                length_penalty=0.0,
                                forced_bos_token_id=tokenizer.get_lang_id(lang_tag))
    else:
        output = model.generate(inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                max_length=256,
                                length_penalty=0.0)

    return output.detach().to("cpu")


def eval_one_doc_annotator(model, tokenizer, row, bilingual=True):
    
    full_kws = row.keywords
    full_kws_lower = full_kws.lower()
    test_kws = full_kws.split(" — ") 
    test_kws_lower = full_kws_lower.split(" — ")

    if bilingual:
        lang_tag = row.language + "_XX"
        tokenizer.src_lang = lang_tag
        tokenizer.tgt_lang = lang_tag

    try:
        output = annotate_one_doc(model, tokenizer, test_kws_lower,
                                  lang_tag=tokenizer.src_lang if bilingual else None)
    except:
        print(row)
        print(len(test_kws_lower))
        return full_kws
    parts = []

    for o, skw in zip(output, test_kws_lower):
        striped_skw = skw.strip()
        decoded = re.split("[ -]", tokenizer.decode(o, skip_special_tokens=True))
        sskw = re.split("[ -]", striped_skw)

        pairs = zip(sskw, decoded)
        decoded_kws = " ".join([capitalize(kw, cap) for (kw, cap) in pairs])
        # Lookup positions of - in original sentence and change the corresponding ws.
        dash_idxs = [i for i, c in enumerate(striped_skw) if c == '-']
        decoded_kws = "".join(['-' if i in dash_idxs else c for i, c in enumerate(decoded_kws)])

        parts.append(decoded_kws)

    return " — ".join(parts)


def eval_transformers_annotator():
    data_dir = "~/lexkey/data/"
    model_name = "facebook/bart-base"

    bilingual = "mbart" in model_name
    model = BartForConditionalGeneration.from_pretrained(data_dir + "models/truecasing/en_annotator")
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_csv(data_dir + "classification_test.csv")
    if not bilingual:
        df = df[df.language == 'en']
    df = df.sample(n=100)

    evaluations = df.apply(lambda r: eval_one_doc_annotator(model, tokenizer, r, bilingual=bilingual), axis=1)
    print(evaluations)
    evaluations.to_csv(data_dir + "trucasing_eval.csv")


def run_transformers_annotator(model_path, data_path, output_path, courts=None, quantized=False, ignore_courts=False):
    model_name = "facebook/bart-base"
    bilingual = "mbart" in model_name

    if quantized:
        model = torch.jit.load(model_path)
    else:
        model = BartForConditionalGeneration.from_pretrained(model_path)

    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_df = pd.read_csv(data_path)

    if not bilingual:
        df = full_df[full_df.language == 'en']

    if ignore_courts:
        training_courts = ['csc-scc']
        df_to_truecase = df[~df["decisionTribunal"].isin(training_courts)]
    else:
        courts_to_format = df[df["decisionTribunal"].isin(courts)]
        other_courts = df[~df.index.isin(courts_to_format.index)]
        all_caps = other_courts[other_courts["keywords"].str.split().str.get(0).str.isupper()]
        df_to_truecase = pd.concat([courts_to_format, all_caps])

    evaluations = df_to_truecase.progress_apply(lambda r:
                                                eval_one_doc_annotator(model, tokenizer, r, bilingual=bilingual),
                                                axis=1)

    full_df.loc[full_df.index.isin(evaluations.index), "keywords"] = evaluations

    full_df.to_csv(output_path)


if __name__ == "__main__":

    model_path = "data/models/en_annotator"

    run_transformers_annotator(model_path=model_path,
                               data_path="data/classification_train.csv",
                               output_path="data/classification_train_cased.csv")
