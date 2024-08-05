# -*- coding: utf-8 -*-
"""keyphrase-generation-t5-small-inspec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bb9Dul2adFOYlxh9CwA3pMnBGiZTkEiB
"""

!pip install datasets
!pip install transformers

import datasets as ds

data=ds.load_dataset("midas/inspec")

from datasets import load_dataset
from transformers import AutoTokenizer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART", add_prefix_space=True)

# Dataset parameters
dataset_full_name = "midas/inspec"
dataset_subset = "raw"
dataset_document_column = "document"

keyphrase_sep_token = ";"

def preprocess_keyphrases(text_ids, kp_list):
    kp_order_list = []
    kp_set = set(kp_list)
    text = tokenizer.decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    text = text.lower()
    for kp in kp_set:
        kp = kp.strip()
        kp_index = text.find(kp.lower())
        kp_order_list.append((kp_index, kp))

    kp_order_list.sort()
    present_kp, absent_kp = [], []

    for kp_index, kp in kp_order_list:
        if kp_index < 0:
            absent_kp.append(kp)
        else:
            present_kp.append(kp)
    return present_kp, absent_kp


def preprocess_fuction(samples):
    processed_samples = {"input_ids": [], "attention_mask": [], "labels": []}
    for i, sample in enumerate(samples[dataset_document_column]):
        input_text = " ".join(sample)
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
        )
        present_kp, absent_kp = preprocess_keyphrases(
            text_ids=inputs["input_ids"],
            kp_list=samples["extractive_keyphrases"][i]
            + samples["abstractive_keyphrases"][i],
        )
        keyphrases = present_kp
        keyphrases += absent_kp

        target_text = f" {keyphrase_sep_token} ".join(keyphrases)

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_text, max_length=40, padding="max_length", truncation=True
            )
            targets["input_ids"] = [
                (t if t != tokenizer.pad_token_id else -100)
                for t in targets["input_ids"]
            ]
        for key in inputs.keys():
            processed_samples[key].append(inputs[key])
        processed_samples["labels"].append(targets["input_ids"])
    return processed_samples

# Load dataset
dataset = load_dataset(dataset_full_name, dataset_subset)
# Preprocess dataset
tokenized_dataset = dataset.map(preprocess_fuction, batched=True)

train_document=[" ".join(i['document']) for i in train]
train_keyphrase=[i['extractive_keyphrases'] for i in train]
train_tags=train['doc_bio_tags']


test_document=[" ".join(i['document']) for i in test]
test_keyphrase=[i['extractive_keyphrases'] for i in test]
test_tags=test['doc_bio_tags']

val_document=[" ".join(i['document']) for i in val]
val_keyphrase=[i['extractive_keyphrases'] for i in val]
val_tags=val['doc_bio_tags']

# Model parameters
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]


# Load pipeline
model_name = "ml6team/keyphrase-generation-t5-small-inspec"
generator = KeyphraseGenerationPipeline(model=model_name)

keyphrases = generator(test_document)

print(keyphrases)

from sklearn.metrics import classification_report

def tag_bio(text,ann):
    
    tags=['O']*len(text)
    last_i=0
    for key_phrase in ann:
        
        for i in range(len(text)):
            
            if tags[i]=='O':
                c=0
                for count in range(len(key_phrase)):
                    if i+count<len(text):
                        if key_phrase[count]!= text[i+count]:
                            break
                        else:
                            c=c+1
                
                if c == len(key_phrase):
                    tags[i]='B'
                    for k in range(1,len(key_phrase)):
                        tags[i+k]='I'
                    last_i=i+c 
    return tags
        

def get_lower(text,ann):
    
    return [[j.lower() for j in i.split()] for i in ann],[i.lower() for i in text.split()]

def preprocess (text,ann):
    return tag_bio(text,ann)

y_pred,y_true=[],[]
for i in range(len(test_document)):
    key,text=get_lower(test_document[i],keyphrases[i])
    
    tag=preprocess(text,key)
    y_pred+=tag
    y_true+=test_tags[i]
    
print(classification_report(y_true,y_pred))

