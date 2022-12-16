"""
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
"""

import sqlite3
import pandas as pd
import numpy as np

import torch
import torch.utils.data
import transformers
import nltk.tokenize.treebank
import sklearn.model_selection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

GEOMETADB_PATH = "/net/data/GEOmetadb.sqlite"
MODEL_SLUG = "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
#model = transformers.AutoModel.from_pretrained(MODEL_SLUG)

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, tokenizer, max_len):
        self.X = X
        self.Y = np.array(Y)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        text = str(self.X[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.Y[index], dtype=torch.float)
        }


def preprocess(value):
    value = value.lower()
    value = re.sub(r'[\r\n]+', ' ', value)
    value = re.sub(r'[^\x00-\x7F]+', ' ', value)

    tokenized = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize(value)
    sentence = ' '.join(tokenized)
    sentence = re.sub(r"\s's\b", "'s", sentence)
    return sentence

def dataset():
    cx = sqlite3.connect(GEOMETADB_PATH)

    Y = pd.read_csv("data/txt2onto/gold_standard/GoldStandard_LabelMatrix.csv")
    Y = Y.iloc[:,1:].set_index(["Sample_ID", "Experiment_ID"])
    Y.index.rename(["SampleID", "ExperimentID"], inplace=True)
    Y = (Y == 1).astype(int)

    X = {}
    for sample_id in Y.index.levels[0]:
        c = cx.cursor()
        c.execute("""
            SELECT gsm, gpl, title, description, source_name_ch1, characteristics_ch1
            FROM gsm
            WHERE gsm=?
        """, (sample_id,))
        sample_id, platform_id, *text = next(iter(c))
        text = " ".join(filter(None, text))
        X[sample_id] = text
    X = pd.Series([X[k] for k in Y.index.levels[0]], index=Y.index.levels[0])

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    X_tr, X_te, Y_tr, Y_te = sklearn.model_selection.train_test_split(X, Y, train_size=0.8)
    train = Dataset(X_tr, Y_tr, tokenizer, 512)
    test = Dataset(X_te, Y_te, tokenizer, 512)
    return train, test

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(MODEL_SLUG)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 347)

    def forward(self, ids, mask, token_type_ids):
        o1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = self.l2(o1["pooler_output"])
        o3 = self.l3(o2)
        return o3

def run(train, test):
    train_loader = torch.utils.data.DataLoader(train, **{
        "batch_size": TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0
    })
    test_loader = torch.utils.data.DataLoader(test, **{
        "batch_size": VALID_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0
    })

    model = Model()
    model.to(DEVICE)

    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for i in range(EPOCHS):
        model.train()
        for _, data in enumerate(train_loader, 0):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
        mask = data['mask'].to(DEVICE, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype = torch.long)
        targets = data['targets'].to(DEVICE, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def validation(epoch):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(test_loader, 0):
                ids = data['ids'].to(DEVICE, dtype = torch.long)
                mask = data['mask'].to(DEVICE, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(DEVICE, dtype = torch.long)
                targets = data['targets'].to(DEVICE, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    for epoch in range(EPOCHS):
        outputs, targets = validation(epoch)
        outputs = np.array(outputs) >= 0.5
        accuracy = sklearn.metrics.accuracy_score(targets, outputs)
        f1_score_micro = sklearn.metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = sklearn.metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
