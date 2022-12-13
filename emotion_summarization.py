from bert_score import score as bert_scr
import pandas as pd
from absl import app
from absl import flags

from datasets import load_metric

import json
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq, BartForConditionalGeneration
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("emotion", "anger", "Emotion type")
flags.DEFINE_string("training_path",
                    "data/train_val_test/train_anonymized_post.json",
                    "Path to training json")
flags.DEFINE_string("validation_path",
                    "data/train_val_test/val_anonymized_post.json",
                    "Path to validation json")
flags.DEFINE_string("test_path",
                    "data/train_val_test/test_anonymized_post.json",
                    "Path to test json")
flags.DEFINE_string("model", "facebook/bart-large-cnn", "Model ")
flags.DEFINE_string("results_summarization", "summarization", "")
flags.DEFINE_integer("batch_size", 2, "")
flags.DEFINE_integer("gradient_accumulation_steps", 8, "")
flags.DEFINE_float("learning_rate", 0.00005, "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rouge_metric = load_metric("rouge")
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def load_split_to_dataset(filepath, emotion):
    with open(filepath) as f:
        dataset = json.load(f)

    positive_posts = []
    summarizations = []

    for k in dataset:
        emo = 0
        summ = []
        for annotation in dataset[k]['Annotations']:
            for hit in dataset[k]['Annotations'][annotation]:
                if hit['Emotion'] == emotion:
                    emo = 1
                    summ.append(hit['Abstractive'])
        if emo == 0:
            continue
        else:
            for i in range(len(summ)):
                positive_posts.append(dataset[k]['Post'])
                summarizations.append(summ[i])

    return positive_posts, summarizations


def load_test_split_to_dataset(filepath, emotion):

    with open(filepath) as f:
        dataset = json.load(f)

    positive_posts = []
    summarizations = []

    for k in dataset:
        emo = 0
        summ = []
        for annotation in dataset[k]['Annotations']:
            for hit in dataset[k]['Annotations'][annotation]:
                if hit['Emotion'] == emotion:
                    emo = 1
                    summ.append(hit['Abstractive'])
        if emo == 0:
            pass
        else:
            positive_posts.append(dataset[k]['Post'])
            summarizations.append(summ)

    return positive_posts, summarizations


def convert_positive_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                truncation=True,
                                padding='max_length')

    target_encodings = tokenizer(text_target=example_batch['Summaries'],
                                 max_length=128,
                                 truncation=True)

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }


def convert_positive_test_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                truncation=True,
                                padding='max_length')

    for i in range(len(example_batch['Summary2'])):
        if example_batch['Summary2'][i] == None:
            example_batch['Summary2'][i] = ''

    target_summaries1 = tokenizer(text_target=example_batch['Summary1'],
                                  max_length=128,
                                  truncation=True,
                                  padding='max_length')
    target_summaries2 = tokenizer(text_target=example_batch['Summary2'],
                                  max_length=128,
                                  truncation=True,
                                  padding='max_length')
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "summary1": target_summaries1["input_ids"],
        "summary2": target_summaries2["input_ids"]
    }


def convert_negative_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["Posts"],
                                max_length=512,
                                padding='max_length',
                                truncation=True)

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"]
    }


def evaluate_summaries(dataset,
                       metric,
                       model,
                       tokenizer,
                       batch_size=8,
                       device=device,
                       column_text="Posts"):

    article_batches = list(chunks(dataset[column_text], batch_size))

    target_batches_s1 = list(chunks(dataset['Summary1'], batch_size))
    target_batches_s2 = list(chunks(dataset['Summary2'], batch_size))
    zped = list(zip(target_batches_s1, target_batches_s2))
    fin_zip = []
    for elem in zped:
        inner_zip = list(zip(elem[0], elem[1]))
        for i in range(len(inner_zip)):
            if inner_zip[i][1] == '':
                inner_zip[i] = [inner_zip[i][0]]
            else:
                inner_zip[i] = list(inner_zip[i])
        fin_zip.append(inner_zip)

    our_summaries = []
    target_summaries = []
    q = False
    for article_batch, target_batch in tqdm(zip(article_batches, fin_zip),
                                            total=len(article_batches)):

        inputs = tokenizer(article_batch,
                           truncation=True,
                           padding="max_length",
                           return_tensors="pt")

        summaries = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            length_penalty=0.8,
            num_beams=8,
            max_length=128)

        decoded_summaries = [
            tokenizer.decode(s,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for s in summaries
        ]
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

        our_summaries += decoded_summaries
        target_summaries += target_batch

        metric.add_batch(predictions=decoded_summaries,
                         references=target_batch)

    _, _, F1 = bert_scr(our_summaries,
                        target_summaries,
                        model_type='microsoft/deberta-xlarge-mnli',
                        lang='en',
                        verbose=True)

    new_score = metric.compute()

    return new_score['rougeL'].mid.fmeasure, F1.mean()


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i + batch_size]


def prepare_dataset(positive_posts, summarizations):
    df_positive = pd.DataFrame(list(zip(positive_posts, summarizations)),
                               columns=['Posts', 'Summaries'])
    df_positive = Dataset.from_pandas(df_positive)
    df_positive = df_positive.map(convert_positive_examples_to_features,
                                  batched=True)
    df_positive.set_format(type="torch",
                           columns=["input_ids", "labels", "attention_mask"])

    return df_positive


def prepare_test_dataset(positive_posts, summarizations):
    s1, s2 = [], []
    for elem in summarizations:
        if len(elem) == 2:
            s1.append(elem[0])
            s2.append(elem[1])
        else:
            s1.append(elem[0])
            s2.append(None)
    df_positive = pd.DataFrame(list(zip(positive_posts, s1, s2)),
                               columns=['Posts', 'Summary1', 'Summary2'])
    df_positive = Dataset.from_pandas(df_positive)
    df_positive = df_positive.map(convert_positive_test_examples_to_features,
                                  batched=True)
    df_positive.set_format(
        type="torch",
        columns=["input_ids", "summary1", "summary2", "attention_mask"])
    return df_positive


def return_dataloaders(model, df_positive):
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                   model=model,
                                                   padding='longest')
    train_positives = torch.utils.data.DataLoader(
        df_positive,
        batch_size=FLAGS.batch_size,
        collate_fn=seq2seq_data_collator,
        num_workers=4,
        shuffle=True)
    return train_positives


def return_test_dataloaders(df_positive):
    train_positives = torch.utils.data.DataLoader(df_positive,
                                                  batch_size=FLAGS.batch_size,
                                                  num_workers=4,
                                                  shuffle=False)

    return train_positives


def ev_once(model, df_positive):
    model.eval()
    rL, bert_score = evaluate_summaries(df_positive,
                                        rouge_metric,
                                        model,
                                        tokenizer,
                                        batch_size=8,
                                        device=device)
    model.train()

    return rL, bert_score


def full_eval(model, df_test_positive, df_validation_positive, results, epoch):
    rL, bert_score = ev_once(model, df_test_positive)
    results[str(epoch) + '_test'] = (rL, float(bert_score.numpy()))
    rL, bert_score = ev_once(model, df_validation_positive)
    results[str(epoch) + '_validation'] = (rL, float(bert_score.numpy()))
    with open(FLAGS.results_summarization + '_' + FLAGS.emotion + '.json',
              'w') as f:
        json.dump(results, f)
    return rL, bert_score


def main(argv):

    global tokenizer

    train_positive_posts, train_summarizations = load_split_to_dataset(
        FLAGS.training_path, FLAGS.emotion)
    dev_positive_posts, dev_summarizations = load_test_split_to_dataset(
        FLAGS.validation_path, FLAGS.emotion)
    test_positive_posts, test_summarizations = load_test_split_to_dataset(
        FLAGS.test_path, FLAGS.emotion)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)

    df_train_positive = prepare_dataset(train_positive_posts,
                                        train_summarizations)
    df_validation_positive = prepare_test_dataset(dev_positive_posts,
                                                  dev_summarizations)
    df_test_positive = prepare_test_dataset(test_positive_posts,
                                            test_summarizations)

    model = BartForConditionalGeneration.from_pretrained(FLAGS.model)
    model.to(device)

    train_dataloader_positives = return_dataloaders(model, df_train_positive)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    results = {}
    full_eval(model, df_test_positive, df_validation_positive, results, 0)
    for epoch in range(10):
        ctr = 1
        for data_positive in tqdm(train_dataloader_positives):
            cuda_tensors_positives = {
                key: data_positive[key].to(device)
                for key in data_positive
            }
            seq2seqlmoutput = model(
                input_ids=cuda_tensors_positives['input_ids'],
                attention_mask=cuda_tensors_positives['attention_mask'],
                labels=cuda_tensors_positives['labels'])
            seq2seqlmoutput.loss.backward()
            if ctr % FLAGS.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            ctr += 1
        optimizer.step()
        optimizer.zero_grad()
        full_eval(model, df_test_positive, df_validation_positive, results,
                  epoch + 1)


if __name__ == "__main__":
    app.run(main)
