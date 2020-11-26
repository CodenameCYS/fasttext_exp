import os
import pickle
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

class MyDense(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.nn.functional.softmax(self.linear(x), dim=-1)

class FasttextModel(torch.nn.Module):
    def __init__(self, vocab_size, label_num, dim=100):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, dim)
        # self.dense_layer = MyDense(dim, label_num)
        self.dense_layer = torch.nn.Linear(dim, label_num)
        
    def forward(self, inputs):
        # print(inputs.shape)
        m = self.embedding_layer(inputs)
        # print(m.shape)
        m = torch.mean(m, dim=1)
        # print("m", m.shape)
        logits = self.dense_layer(m)
        # print("logits", logits.shape)
        return logits

def cross_entropy(y_pred, y_true):
    # y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
    category = y_pred.size()[-1]
    y_true = torch.nn.functional.one_hot(y_true, num_classes=category)
    loss = - y_true * torch.log(y_pred)
    return torch.mean(torch.sum(loss, dim=-1))

def train_model(x, y, vocab_size, label_num, epochs=20, dim=300, batch_size=1024):
    model = FasttextModel(vocab_size, label_num, dim)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = cross_entropy

    for epoch in range(epochs):
        with tqdm(range((len(x)-1) // batch_size + 1), ncols=100) as t:
            for i in t:
                optimizer.zero_grad()
                y_pred = model(torch.tensor(x[i*batch_size:(i+1)*batch_size]))
                y_true = torch.tensor(y[i*batch_size:(i+1)*batch_size])
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
        
        print("train {} epochs, loss = {}".format(epoch+1, loss.item()))
    
    return model

def evaluate_model(model, x, y_true):
    y_pred = torch.argmax(model(torch.tensor(x)), axis=-1).tolist()
    report = classification_report(y_true, y_pred)
    return report

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    test_src = pickle.load(open("data/tensorflow/test.src.pkl", "rb"))
    test_tgt = pickle.load(open("data/tensorflow/test.tgt.pkl", "rb"))
    train_src = pickle.load(open("data/tensorflow/train.src.pkl", "rb"))
    train_tgt = pickle.load(open("data/tensorflow/train.tgt.pkl", "rb"))

    vocab_size = len([line.strip() for line in open("data/tensorflow/vocab.txt")])
    label_num = len([line.strip() for line in open("data/tensorflow/label.txt")])
    model = train_model(train_src, train_tgt, vocab_size, label_num, epochs=50, dim=300, batch_size=1024)
    report = evaluate_model(model, test_src, test_tgt)
    print(report)
    with open("log/pytorch_exp.log", "w+") as fp:
        fp.write(report)