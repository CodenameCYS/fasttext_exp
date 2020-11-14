import fasttext
from sklearn.metrics import classification_report

def evaluate_model(test_file):
    inputs = []
    y_true = []
    for line in open(test_file):
        if line.strip() == "":
            continue
        tmp = line.strip().split("\t")
        label = tmp[0]
        text = "\t".join(tmp[1:])
        y_true.append(label)
        inputs.append(text)
    y_pred = model.predict(inputs)[0]
    report = classification_report(y_true, y_pred)
    return report

if __name__ == "__main__":
    model = fasttext.train_supervised('data/fasttext/train.txt', epoch=50, lr=0.05, dim=300, maxn=128)
    report = evaluate_model('data/fasttext/test.txt')
    print(report)
    with open("log/fasttext_exp.log", "w+") as fp:
        fp.write(report)