import os
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report

class MyLoss(tf.keras.losses.Loss):
    def __init__(self, label_num):
        super().__init__()
        self.label_num = label_num
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # print(y_true.shape)
        y_true = tf.one_hot(tf.squeeze(y_true, axis=1), self.label_num)
        # print(y_true.shape)
        return self.cross_entropy(y_true, y_pred)

class FasttextModel(tf.keras.Model):
    def __init__(self, vocab_size, label_num, dim=100):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, dim)
        self.dense_layer = tf.keras.layers.Dense(label_num, activation=tf.nn.softmax)
        
    def call(self, inputs, training=False):
        # print(inputs.shape)
        m = self.embedding_layer(inputs)
        # print(m.shape)
        m = tf.math.reduce_mean(m, axis=1)
        # print("m", m.shape)
        # logits = tf.nn.softmax(self.dense_layer(m), axis=-1)
        logits = self.dense_layer(m)
        # print("logits", logits.shape)
        
        if training:
            return logits
        else:
            return {
                "tag": tf.math.argmax(logits, axis=-1), 
                "prob": tf.math.reduce_max(logits, axis=-1)
            }

def evaluate_model(model, x, y_true):
    y_pred = model.predict(x)["tag"]
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
    model = FasttextModel(vocab_size=vocab_size, label_num=label_num, dim=300)
    model.compile(optimizer='adam', loss=MyLoss(label_num))
    model.fit(x=train_src, y=train_tgt, batch_size=1024, epochs=50)
    report = evaluate_model(model, test_src, test_tgt)
    print(report)
    with open("log/tensorflow_exp.log", "w+") as fp:
        fp.write(report)