import pandas as pd
from absl import app
from absl import flags
from data import load_split_to_dataset
from sklearn.metrics import f1_score

import json
import torch

from data import EmotionsDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("emotion", "anger", "Emotion type")
flags.DEFINE_string("training_path", "data/train_val_test/train_anonymized_post.json", "Path to training json")
flags.DEFINE_string("validation_path", "data/train_val_test/val_anonymized_post.json", "Path to validation json")
flags.DEFINE_string("test_path", "data/train_val_test/test_anonymized_post.json", "Path to test json")
flags.DEFINE_string("model", "bert-large-uncased", "Model ")
flags.DEFINE_integer("batch_size", 8, "")
flags.DEFINE_integer("gradient_accumulation_steps", 2, "")
flags.DEFINE_string("results_detection", "detection", "")
flags.DEFINE_float("learning_rate", 0.00005, "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels_list, tokenizer):
        self.text_list = text_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        tok_post = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=512, truncation=True)
        tok_post['label'] = self.labels_list[idx]
        item = {key: torch.tensor(tok_post[key]) for key in tok_post if key != 'label'}
        item['label'] = torch.tensor(tok_post['label'], dtype=float)
        return item

    def __len__(self):
        return len(self.text_list)


class EmotionModel(torch.nn.Module):
    def __init__(self, ckpt_file):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(ckpt_file)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(1024, 1)

    def forward(self, x):
        out = self.bert_model(
            x['input_ids'], x['attention_mask']).last_hidden_state[:, 0, :]
        out = torch.squeeze(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def load_split_to_dataset(filepath, emotion):

    with open(filepath) as f:
        dataset = json.load(f)

    positive_posts = [] 
    negative_posts = []

    for k in dataset:
        emo = 0
        summ = []
        for annotation in dataset[k]['Annotations']:
            for hit in dataset[k]['Annotations'][annotation]:
                if hit['Emotion'] == emotion:
                    emo = 1
                    summ.append(hit['Abstractive'])
        if emo == 0:
            negative_posts.append(dataset[k]['Post'])
        else:
            positive_posts.append(dataset[k]['Post'])

    return positive_posts, negative_posts


def evaluate_detection(model, test_dataloader):
    full_predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                for key in elem if key not in ['text', 'idx']}
            logits = model(x)
            results = torch.where(logits > 0, 1, 0)
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + list(elem['label'].cpu().detach().numpy())

    model.train()

    return str(f1_score(true_labels, full_predictions, average=None))


def main(argv):

    train_positive_posts, train_negative_posts = load_split_to_dataset(FLAGS.training_path, FLAGS.emotion)
    validation_positive_posts, validation_negative_posts = load_split_to_dataset(FLAGS.validation_path, FLAGS.emotion)
    test_positive_posts, test_negative_posts = load_split_to_dataset(FLAGS.test_path, FLAGS.emotion)

    train_posts, train_emotions = train_positive_posts + train_negative_posts, [1] * len(train_positive_posts) + [0] * len(train_negative_posts)
    validation_posts, validation_emotions = validation_positive_posts + validation_negative_posts, [1] * len(validation_positive_posts) + [0] * len(validation_negative_posts)
    test_posts, test_emotions = test_positive_posts + test_negative_posts, [1] * len(test_positive_posts) + [0] * len(test_negative_posts)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)

    training_dataset = EmotionsDataset(train_posts, train_emotions, tokenizer)
    validation_dataset = EmotionsDataset(validation_posts, validation_emotions, tokenizer)
    test_dataset = EmotionsDataset(test_posts, test_emotions, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True)


    model = EmotionModel(FLAGS.model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    f1_validation = evaluate_detection(model, validation_dataloader)
    f1_test = evaluate_detection(model, test_dataloader)
    results = {}
    results['0_test'] = f1_test
    results['0_validation'] = f1_validation
    for epoch in range(10):
        crt = 1
        for data in tqdm(train_dataloader):
            cuda_tensors = {key: data[key].to(
                device) for key in data if key not in ['text', 'idx']}

            
            logits = model(cuda_tensors)
            loss = loss_fn(torch.squeeze(logits), cuda_tensors['label'])
            loss.backward()
            if crt % FLAGS.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            crt += 1
        f1_validation = evaluate_detection(model, validation_dataloader)
        f1_test = evaluate_detection(model, test_dataloader)
        results[str(epoch + 1) + '_test'] = f1_test
        results[str(epoch + 1) + '_validation'] = f1_validation
        with open(FLAGS.results_detection + '_' + FLAGS.emotion + '.json', 'w') as f:
            json.dump(results, f)
        
if __name__ == "__main__":
    app.run(main)
