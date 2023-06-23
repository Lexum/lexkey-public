import pandas as pd
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
from tqdm import tqdm
import re
import os
import nltk
from truecasing import run_transformers_annotator
tqdm.pandas()


def get_text_from_html(html):
    soup = BeautifulSoup(html, features="html.parser") 
    try:
        doc_content = soup.find("body").get_text()
    except AttributeError:
        doc_content = soup.get_text()
         
    return doc_content


def preprocess_text(text):
    # TODO also keep list elements
    text = text.replace("&nbsp", " ")
    text = text.replace(u'\xa0', " ")
    text = re.sub('__+', '__', text)
    text = re.sub('\*\*+', '\*\*', text)
    text = re.sub('  +', ' ', text)
    text = re.sub('\n \n', '\n', text)
    return text.replace('\n', '</n>')


def convert_and_save_one_doc(filename, encoding, data_dir, output_dir):
    # load the html from filename and convert it to text
    output_name = filename.split(".")[0]
    output_path = output_dir + output_name + ".txt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_path) and os.path.isfile(output_path):
        text = get_text_from_file(output_path)
    else:
        try:
            if encoding == 'WINDOWS_LATIN_1':
                encoding = 'WINDOWS-1252'
            with open(data_dir + filename, "r", encoding=encoding) as f:
                text = get_text_from_html(f.read())
        except Exception as e:
            print("Could not convert " + filename)
            print(e)
            return 0, 0, filename.replace(".html", ".txt")

        text = preprocess_text(text)
        with open(output_path, "w", encoding='utf-8') as of:
            of.write(text)

    return len(text), len(text.split()), filename.replace(".html", ".txt")

def get_text_from_file(path):
    with open(path, "r", encoding='utf-8') as f:
        return f.read()

def step3_extract_html(data_dir, meta_path, output_dir):
    # output all html converted to text to step3 dir
    # Get them from noKeywordsFilename on cerebro
    manifest = load_metadatas(meta_path)
    print(manifest.iloc[0].filename)
    # create dir for output
    Path(output_dir).mkdir(exist_ok=True)
    
    # for each document, convert the html to text and save to output_dir
    # Update manifest with char length and word length of documents
    manifest["Char Length"] = manifest["Word Length"] = None
    manifest[["Char Length", "Word Length", "filename"]] = manifest.progress_apply(
        lambda one_doc: convert_and_save_one_doc(one_doc.filename, one_doc.encoding, data_dir, output_dir),
        axis=1, result_type="expand")
    manifest.to_csv(output_dir + "manifest_sizes.csv")


def hash_dataframe_ids(df, proportion):
    num_buckets = int(1/proportion)
    return df.Id.map(lambda r: int(hashlib.md5(r.encode('utf-8')).hexdigest(), 16) % num_buckets)


def split_by_hashes(df, proportion):
    buckets = hash_dataframe_ids(df, proportion)
    train = df[buckets > 1]
    valid = df[buckets == 0]
    test = df[buckets == 1]
    return train, valid, test


def split_dataframe(df, proportion):
    return split_by_hashes(df, proportion)


def load_metadatas(path):
    metas = pd.read_csv(path)
    return metas


def merge_datasets(df1, df2):
    # Check if index are ok
    return pd.concat([df1, df2])


def separate_documents_on_missing(df, columns):
    labeled = df.dropna(subset=columns)
    other = df[~df.index.isin(labeled.index)]
    return labeled, other


def separate_documents_with_subjects(df):
    return separate_documents_on_missing(df, ["keywords"])


def trim_missing_translations(en_df, fr_df, id_col, tr_col):
    en_subset = en_df.dropna(subset=[tr_col])
    fr_subset = fr_df.dropna(subset=[tr_col])
    
    translated_en = en_subset[en_subset[tr_col].isin(fr_subset[id_col])]
    translated_fr = fr_subset[fr_subset[tr_col].isin(en_subset[id_col])]
    
    trimmed_en = en_subset[~en_subset.index.isin(translated_en.index)]
    trimmed_fr = fr_subset[~fr_subset.index.isin(translated_fr.index)]
    
    en_diff = len(en_subset) - len(translated_en)
    fr_diff = len(fr_subset) - len(translated_fr)
    print("Trimmed " + str(en_diff) + " english docs and " + str(fr_diff) + " french docs")
    
    return translated_en, translated_fr, pd.concat([trimmed_en, trimmed_fr])
    
    
def separate_documents_with_translations(df, keep_trimmed=False):
    french, english = [], []
    # Get all decisions with translationId
    translated_decisions, rest = separate_documents_on_missing(df, ["decisionTranslationId"])
    print("\nTranslated decisions :" + str(len(translated_decisions)))
    # Split by language
    french_decisions = translated_decisions[translated_decisions["language"] == "fr"]
    english_decisions = translated_decisions[translated_decisions["language"] == "en"]
    french_decisions, english_decisions, trimmed_decisions = trim_missing_translations(english_decisions,
                                                                                       french_decisions,
                                                                                       "cc2DecisionId",
                                                                                       "decisionTranslationId")
    
    french.append(french_decisions)
    english.append(english_decisions)
    
    print("English decisions :" + str(len(english_decisions)))
    print("French decisions :" + str(len(french_decisions)))

    # Group doctrine and legis by id and split by language
    other = rest.dropna(subset=["legislationId", "contributionId"], how="all")
    b_other = other[other.duplicated(subset=["legislationId", "contributionId"], keep=False)]
    
    # Split b_other into french and english
    fr_other = b_other[b_other["language"] == "fr"]
    en_other = b_other[b_other["language"] == "en"]

    en_legis, fr_legis, trimmed_legis = trim_missing_translations(en_other,
                                                                  fr_other,
                                                                  "legislationId",
                                                                  "legislationId")
    en_doctrine, fr_doctrine, trimmed_doctrine = trim_missing_translations(en_other,
                                                                           fr_other,
                                                                           "contributionId",
                                                                           "contributionId")
    
    print("English legis " + str(len(en_legis)))
    print("French legis " + str(len(fr_legis)))
    print("English doctrine " + str(len(en_doctrine)))
    print("French doctrine " + str(len(fr_doctrine)))
    
    french.append(fr_legis)
    french.append(fr_doctrine)
    english.append(en_legis)
    english.append(en_doctrine)

    rest = rest[~rest.index.isin(b_other.index)]

    if keep_trimmed:
        pd.concat([rest, trimmed_decisions, trimmed_legis, trimmed_doctrine])
        
    return pd.concat(french), pd.concat(english), rest


def get_all_juris_id(df):
    return df["jurisdiction"].unique()


def append_translations(df, all_translated):
    print("English: " + str(len(df)))
    
    all_decisions = all_translated.dropna(subset=["decisionTranslationId"])
    translated_decisions = all_decisions[all_decisions["decisionTranslationId"].isin(df["cc2DecisionId"])]
    print("French decisions: " + str(len(translated_decisions)))
    
    all_legis = all_translated.dropna(subset=["legislationId"])
    translated_legis = all_legis[all_legis["legislationId"].isin(df["legislationId"])]
    print("French legis: " + str(len(translated_legis)))
    
    all_doctrine = all_translated.dropna(subset=["contributionId"]) 
    translated_doctrine = all_doctrine[all_doctrine["contributionId"].isin(df["contributionId"])]
    print("French doctrine: " + str(len(translated_doctrine)))
    
    print("French total:" + str(len(translated_decisions) + len(translated_legis) + len(translated_doctrine)))
        
    return pd.concat([df, translated_decisions, translated_legis, translated_doctrine])


def split_documents_with_subjects(meta_df, test_proportion):
    # Separate decisions with subjects
    # TODO make sure translated decisions end up in same dataset?
    labeled, unlabeled = separate_documents_with_subjects(meta_df)
    
    french, english, monolingual = separate_documents_with_translations(labeled)
    labeled = pd.concat([english, monolingual])
    
    # For each juris_id, Split decisions with subjects
    # If they are bilingual, make sure to include both if appropriate.
    all_labeled_juris = get_all_juris_id(labeled)
    trs, vals, tes = [], [], [] 
    for juris in all_labeled_juris:
        juris_df = labeled[labeled["jurisdiction"] == juris]
        train, valid, test = split_dataframe(juris_df, test_proportion)
        trs.append(train)
        vals.append(valid)
        tes.append(test)

    # Merge all split juris ids into classification dataset
    b_train = pd.concat(trs)
    b_valid = pd.concat(vals)
    b_test = pd.concat(tes)
    
    # Split french decisions in correct set
    classification_train = append_translations(b_train, french)
    classification_valid = append_translations(b_valid, french)
    classification_test = append_translations(b_test, french)
    
    return classification_train, classification_valid, classification_test, unlabeled


def split_translated_documents(unlabeled, test_proportion):
    # Separate bilingual documents without subjects
    french, english, monolingual = separate_documents_with_translations(unlabeled, keep_trimmed=True)
    
    print("French total: " + str(len(french)))
    print("English total: " + str(len(english)))
    
    # Split english version of bilingual documents into translation dataset
    e_train, e_valid, e_test = split_dataframe(english, test_proportion)
    
    # append corresponding french versions to bilingual dataset
    b_train = append_translations(e_train, french)
    b_valid = append_translations(e_valid, french)
    b_test = append_translations(e_test, french)
    
    return b_train, b_valid, b_test, monolingual


def split_other_documents(monolingual, test_proportion):
    # For each juris_id, split the other documents
    all_juris = get_all_juris_id(monolingual)
    trs, vals, tes = [], [], [] 
    for juris in all_juris:
        juris_df = monolingual[monolingual["jurisdiction"] == juris]
        train, valid, test = split_dataframe(juris_df, test_proportion)
        trs.append(train)
        vals.append(valid)
        tes.append(test)
    return trs, vals, tes


def get_trimmed_labeled_translation(df, monolingual):
    return append_translations(df, monolingual)


def step4_split_dataset(meta_path, test_proportion, output_dir, classification_only=False, translation_only=False):
    # Load metadata csv
    meta_df = load_metadatas(meta_path)
    print(meta_df.columns)

    if not translation_only:
        print("\nSplit classification\n")
        classification_train, classification_valid, classification_test, _ = split_documents_with_subjects(meta_df, test_proportion)
        # Write all datasets to output_dir
        classification_train.to_csv(output_dir + "/classification_train.csv")
        classification_valid.to_csv(output_dir + "/classification_valid.csv")
        classification_test.to_csv(output_dir + "/classification_test.csv")

    if not classification_only:
        print("\nSplit translation\n")
        translation_train, translation_valid, translation_test, monolingual = split_translated_documents(meta_df, test_proportion)
        translation_train.to_csv(output_dir + "/translation_train.csv")
        translation_valid.to_csv(output_dir + "/translation_valid.csv")
        translation_test.to_csv(output_dir + "/translation_test.csv")

        if not translation_only:
            print("\nSplit global\n")
            trs, vals, tes = split_other_documents(monolingual, test_proportion)
        
            # Add all to global dataset
            global_train = pd.concat(trs + [translation_train])
            global_valid = pd.concat(vals + [translation_valid])
            global_test = pd.concat(tes + [translation_test])
            total_len = len(global_train) + len(global_valid) + len(global_test)
            print("Total : " + str(total_len) + ", Starting : " + str(len(meta_df)))

            global_train.to_csv(output_dir + "/global_train.csv")
            global_valid.to_csv(output_dir + "/global_valid.csv")
            global_test.to_csv(output_dir + "/global_test.csv")


def get_text(data_dir, filename):
    with open(data_dir + filename, "r", encoding='utf-8') as f:
        return f.read()


def get_cut_indexes(row, data_dir, max_length, avg_token_char_len, min_word_count):
    full_text = get_text(data_dir, row.filename)
    
    language = 'english' if row.language == 'en' else 'french'
    sents = nltk.tokenize.sent_tokenize(full_text, language=language)
    chunk_size = max_length * avg_token_char_len
    min_size = min_word_count * avg_token_char_len
    cuts = [0]
    block_length = 0
    total_length = 0
    
    for s in sents :
        length = len(s) + 1
        block_length += length
        total_length += length
        if (len(full_text) - total_length) < min_size :
            break
        if block_length >= chunk_size:
            block_length = 0
            cuts.append(total_length)

    cuts.append(len(full_text)) 
    # Zip cuts as start-end pairs
    cuts_zip = list(zip(cuts[:-1], cuts[1:]))
    return cuts_zip


def split_large_doc(df, data_dir, max_length=1024, avg_token_char_len=16, min_word_count=10):

    # Change this to split along sentences instead of chunk_size.
    df["Offset"] = df.progress_apply(
        lambda row: get_cut_indexes(row, data_dir, max_length, avg_token_char_len, min_word_count),
        axis=1)
    df = df.explode("Offset", ignore_index=True)
    df[["Start pos", "End pos"]] = df.apply(lambda t: (t["Offset"][0], t["Offset"][1]), axis=1, result_type="expand")
    return df


def split_docs_in_dataset(data_dir, df_path):
    df = pd.read_csv(df_path)
    df = split_large_doc(df, data_dir)
    df.to_csv(df_path)


def step5_split_large_docs(output_dir):
    text_dir = output_dir + "texts"
    split_docs_in_dataset(text_dir, output_dir + "/global_train.csv")
    split_docs_in_dataset(text_dir, output_dir + "/global_valid.csv")
    split_docs_in_dataset(text_dir, output_dir + "/global_test.csv")   


def step6_truecase_classification(data_path, model_path, quantized=True):

    train_path = data_path + "classification_train.csv"
    valid_path = data_path + "classification_valid.csv"
    test_path = data_path + "classification_test.csv"

    train_output_path = data_path + "truecased_classification_train.csv"
    valid_output_path = data_path + "truecased_classification_valid.csv"
    test_output_path = data_path + "truecased_classification_test.csv"

    run_transformers_annotator(model_path=model_path,
                               data_path=train_path,
                               output_path=train_output_path,
                               quantized=quantized,
                               ignore_courts=True)
    run_transformers_annotator(model_path=model_path,
                               data_path=valid_path,
                               output_path=valid_output_path,
                               quantized=quantized,
                               ignore_courts=True)
    run_transformers_annotator(model_path=model_path,
                               data_path=test_path,
                               output_path=test_output_path,
                               quantized=quantized,
                               ignore_courts=True)


def main():
    data_dir = "~/lexkey/data/"
    output_dir = "~/lexkey/data/"
    meta_path = data_dir + "/merged-manifest.csv"
    meta_path_sizes = output_dir + "texts/manifest_sizes.csv"

    step3_extract_html(data_dir, meta_path, output_dir + "texts/")
    step4_split_dataset(meta_path_sizes, 0.05, output_dir)
    step5_split_large_docs(output_dir)
    step6_truecase_classification(data_dir, data_dir + "models/en_annotator", quantized=False)


if __name__ == "__main__":
    main()
