from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nltk
from transformers.data.data_collator import DataCollatorForLanguageModeling


class LexLMDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 data_dir,
                 bilingual=False,
                 need_lang_tag=False,
                 do_sent_sub=True,
                 max_length=1024,
                 sample=None,
                 split_doc=False):
        
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_dir = data_dir + "texts"
        self.need_lang_tag = need_lang_tag
        self.do_sent_sub = do_sent_sub
        self.max_length = max_length
        self.split_doc = split_doc
        self.token_char_length = 16
        self.min_word_count = 10
        
        if data_path is not None:
            self.dataframe = pd.read_csv(data_path)
            self.dataframe = self.dataframe[self.dataframe["Word Length"] > self.min_word_count]
            if not bilingual:
                self.dataframe = self.dataframe[self.dataframe.language == 'en']
            if sample is not None:
                self.dataframe = self.dataframe.sample(frac=sample)
            if self.split_doc: 
                self.split_large_doc()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = self.get_text(row.filename.replace(".html", ".txt"))
        
        sents = self.split_sentences(text, row)
        sents = self.trim_sentences(sents)
        text = " ".join(sents)
        toks = self.tokenize(text, row.language)
        
        if self.do_sent_sub:
            shuf_text = self.permutate_sentences(sents)
            shuf_toks = self.tokenize(shuf_text, row.language)
            toks["shuffled_ids"] = shuf_toks.input_ids
        
        if len(text) == 1:
            print("Index : " + str(idx))
            
        return toks

    def split_large_doc(self):
        chunk_size = self.max_length * self.token_char_length
        # Create list of split points in doc by chunk size
        # Trim ends too short to meet min word count
        self.dataframe["Offset"] = self.dataframe["Char Length"]\
            .map(lambda x: [s for s in range(0, x, chunk_size)
                            if (s == 0) or ((x - s - chunk_size) > (self.min_word_count * self.token_char_length))])
        # Explode df along list of splits
        self.dataframe = self.dataframe.explode("Offset", ignore_index=True)

    def get_text(self, filename):
        with open(self.data_dir + filename, "r", encoding='utf-8') as f:
            return f.read()

    def set_lang(self, text, lang):
        lang_tag = lang + "_XX" 
        if "mbart" in str(type(self.tokenizer)):
            self.tokenizer.src_lang = lang_tag
            self.tokenizer.tgt_lang = lang_tag
            return text
        else:
            return lang_tag + " " + text 

    def tokenize(self, text, lang):
        if self.need_lang_tag:
            text = self.set_lang(text, lang)
        tokens = self.tokenizer(text,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length,
                                return_special_tokens_mask=True)
        return tokens

    def get_start_end_positions(self, row):
        offset = row.Offset if ('Offset' in self.dataframe.columns) else 0
        start = row['Start pos'] if ('Start pos' in self.dataframe.columns) else None
        end = row['End pos'] if ('End pos' in self.dataframe.columns) else None
        
        if start is None:
            if type(offset) is tuple:
                start = offset[0]
            else:
                start = offset
                
        if end is None:
            if type(offset) is tuple:
                end = offset[-1]
            else:
                end = offset + self.max_length*self.token_char_length
        return start, end

    def split_sentences(self, text, row):                    
        language = 'english' if row.language == 'en' else 'french'
        start, end = self.get_start_end_positions(row)
        trimmed_text = text[start:end]
        return nltk.tokenize.sent_tokenize(trimmed_text, language=language)
        
    def trim_sentences(self, sents):
        total = 0
        trimmed_sents = []
        for s in sents:
            s_toks = self.tokenizer(s).input_ids
            s_len = len(s_toks)
            if total + s_len <= self.max_length:
                total += s_len
                trimmed_sents.append(s)
            else:
                remainer = (self.max_length - total)
                total += remainer
                if remainer > 0:
                    ns = "".join(self.tokenizer.decode(s_toks[:remainer], skip_special_tokens=True))
                    trimmed_sents.append(ns)
                break
            
        return trimmed_sents if total > 0 else [""]

    @staticmethod
    def permutate_sentences(sents):
        np.random.shuffle(sents)
        shuffled_text = " ".join(sents)
        return shuffled_text


class DenoisingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, do_sent_sub=True):
        self.do_sent_sub = do_sent_sub
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.35)
        self.label_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
    def __call__(self, inputs):
        
        if not self.do_sent_sub:
            return self.mlm_collator(inputs)
        
        # Preprocess inputs to shuffle sentences

        shuffled_inputs = [{"input_ids": i.shuffled_ids, "attention_mask": i.attention_mask} for i in inputs]
        
        ordered_batch = self.label_collator(inputs)
        shuffled_batch = self.mlm_collator(shuffled_inputs)
        
        result = shuffled_batch
        result["labels"] = ordered_batch["labels"]
                
        return result    
