import os
import re
import pandas as pd
from xml.dom import minidom
import torch
import transformers as tf

import const

def get_relations() -> pd.DataFrame:
    platinum_df = pd.read_table('./data/matres/platinum.txt', header=None, sep='\t', names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    # Fixing a typo in the docid
    platinum_df.loc[platinum_df['docid'] == 'nyt_20130321_sarcozy', 'docid'] = 'nyt_20130321_sarkozy'

    aquaint_df = pd.read_table('./data/matres/aquaint.txt', header=None, sep='\t', names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    
    relations_df = pd.concat([platinum_df, aquaint_df], ignore_index=True)
    relations_df[['eiid1', 'eiid2']] = 'E' + relations_df[['eiid1', 'eiid2']].astype(str)

    return relations_df

def get_docs(doc_ids) -> pd.DataFrame:
    def purify_text(text):
        # Strip TIMEX3 attributes
        text = re.sub(r'<TIMEX3\b[^>]*/>', '[TIMEX3/]', text)
        text = re.sub(r'<TIMEX3\b[^>]*>', '[TIMEX3]', text)
        text = re.sub(r'</TIMEX3>', '[/TIMEX3]', text)

        event_stack = []

        # Replace EVENT tags with their eid values, capitalized
        def _replace_event_tag(match):
            tag = match.group(0)

            if tag.startswith('</EVENT'):
                if event_stack:
                    return f"[/{event_stack.pop()}]"
                return tag

            eid_match = re.search(r'\beid\s*=\s*"([^"]+)"', tag)
            if eid_match:
                eid = eid_match.group(1)
                eid = eid.upper()
                event_stack.append(eid)
                return f"[{eid}]"
            return tag

        text = re.sub(r'</EVENT>|<EVENT\b[^>]*>', _replace_event_tag, text)

        return text

    def split_sentences(text):
        sentences = re.split(r'(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<=[.!?])\s+', text)
        return sentences
    
    doc_texts = {}
    for docid in doc_ids:
        filename = f"{docid}.tml"

        tempeval_path = f'./data/tempeval/{filename}'
        aquaint_path = f'./data/aquaint/{filename}'
        
        if os.path.exists(tempeval_path):
            filepath = tempeval_path
        elif os.path.exists(aquaint_path):
            filepath = aquaint_path
        else:
            raise FileNotFoundError(f"File {filename} not found in ./data/tempeval or ./data/aquaint")

        text_content = minidom.parse(filepath).getElementsByTagName('TEXT')[0]
        text_content_str = ''.join(node.toxml() for node in text_content.childNodes).strip()
        text_content_purified = purify_text(text_content_str)
        sentences = split_sentences(text_content_purified)
        doc_texts[docid] = {
            'raw_text': text_content_str,
            'text': text_content_purified,
            'sentences': sentences
        }

    docs_df = pd.DataFrame.from_dict(doc_texts, orient='index').reset_index()
    docs_df.columns = ['docid', 'raw_text', 'text', 'sentences']
    docs_df = docs_df.set_index('docid')

    return docs_df
    
def create_context_window(docs_df: pd.DataFrame, docid: str, eiid1: str, eiid2: str, padding: int = 1) -> str:
    sentences = docs_df.loc[docid, 'sentences']
    sentence_indices = []
    for i, sentence in enumerate(sentences):
        if f'[{eiid1}]' in sentence or f'[{eiid2}]' in sentence:
            sentence_indices.append(i)

    if not sentence_indices:
        return ""

    start_index = max(0, min(sentence_indices) - padding)
    end_index = min(len(sentences), max(sentence_indices) + padding + 1)

    context_window = ' '.join(sentences[start_index:end_index])

    context_window = re.sub(rf'\[{re.escape(eiid1)}\]', '[T1]', context_window)
    context_window = re.sub(rf'\[/{re.escape(eiid1)}\]', '[/T1]', context_window)
    context_window = re.sub(rf'\[{re.escape(eiid2)}\]', '[T2]', context_window)
    context_window = re.sub(rf'\[/{re.escape(eiid2)}\]', '[/T2]', context_window)

    # Remove all other event tags like [E3], [/E3], etc.
    context_window = re.sub(r'\[/?E\d+\]', '', context_window)

    # Clean extra whitespace
    context_window = re.sub(r'\s+', ' ', context_window).strip()

    # context_window = f"{verb1} | {verb2} | {context_window}"
    return context_window

def create_context_windows(relations_df: pd.DataFrame, docs_df: pd.DataFrame, padding: int = 1) -> pd.DataFrame:
    df = relations_df.copy()
    df['context_window'] = df.apply(lambda row: create_context_window(docs_df, row['docid'], row['eiid1'], row['eiid2'], padding), axis=1)
    return df

def create_relation_labels(relations_df: pd.DataFrame) -> pd.DataFrame:
    df = relations_df.copy()
    df['relation_id'] = df['relation'].map(const.relation2id)
    return df

def get_relation_stats(relations_df: pd.DataFrame) -> pd.DataFrame:
    # Summary statistics for relation labels
    relation_counts = relations_df["relation"].value_counts().sort_index()
    relation_percents = relations_df["relation"].value_counts(normalize=True).sort_index().mul(100)

    relation_summary = pd.DataFrame({
        "count": relation_counts,
        "percent": relation_percents.round(2)
    })

    print("Total rows:", len(relations_df))
    print("Unique relations:", relations_df["relation"].nunique())
    return relation_summary

def augment_data(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    mask = df["relation"].isin(["BEFORE", "AFTER", "VAGUE", "EQUAL"])
    swapped_df = df.loc[mask].copy()

    # Swap event pair columns
    swapped_df[["verb1", "verb2"]] = swapped_df[["verb2", "verb1"]].to_numpy()
    swapped_df[["eiid1", "eiid2"]] = swapped_df[["eiid2", "eiid1"]].to_numpy()

    # Invert relation
    swapped_df["relation"] = swapped_df["relation"].map({"BEFORE": "AFTER", "AFTER": "BEFORE", "VAGUE": "VAGUE", "EQUAL": "EQUAL"})

    new_df = pd.concat([df, swapped_df], ignore_index=True)

    if verbose:
        print(f"Added {len(swapped_df)} swapped rows")
        print(f"New total rows: {len(new_df)}")

    return new_df

def create_data_loader(df: pd.DataFrame, tokenizer: tf.AutoTokenizer, batch_size=16):
    encodings = tokenizer(
        df["context_window"].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    labels = torch.tensor(df["relation_id"].values, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        labels
    )

    t1_id = tokenizer.convert_tokens_to_ids("[T1]")
    t2_id = tokenizer.convert_tokens_to_ids("[T2]")

    ids = encodings["input_ids"][0].tolist()
    assert t1_id in ids and t2_id in ids, "Truncated away a target event!"

    print("input_ids shape:", encodings["input_ids"].shape)
    print("attention_mask shape:", encodings["attention_mask"].shape)
    print("labels shape:", labels.shape)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def decode_ids(tokenizer: tf.AutoTokenizer, token_ids, skip_special_tokens=False) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def batch_decode_ids(tokenizer: tf.AutoTokenizer, batch_token_ids, skip_special_tokens=False) -> list[str]:
    if isinstance(batch_token_ids, torch.Tensor):
        batch_token_ids = batch_token_ids.tolist()
    return tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)