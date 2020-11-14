import os
import re
import math
import numpy
import pickle
from tqdm import tqdm
from random import shuffle

def clean_data(line):
    punctuation = [",","/",";","\"",":","?","!","@","#","$","%","^","&","(",")","_","+","=","~","<",">", "\\"]
    line = line.replace("<br />"," ").strip()
    for p in punctuation:
        line = line.replace(p, " {} ".format(p))
    line = re.sub("[ ]'", " ' ", line)
    line = re.sub("'[ $]", " ' ", line)
    line = re.sub("\. ", " . ", line)
    line = re.sub("(\.)+ \. ", " ... ", line)
    return line

def create_fasttext_dataset(datapath, foutpath, num=math.inf, need_shuffle=True):
    os.makedirs(os.path.split(foutpath)[0], exist_ok=True)
    data = []
    with tqdm(os.listdir(os.path.join(datapath, "neg")), ncols=100) as t:
        for idx, filename in enumerate(t):
            if idx >= num:
                break
            fid, label = os.path.splitext(filename)[0].split("_")
            data.append("__label__{}\t{}".format(label, clean_data(open(os.path.join(datapath, "neg", filename)).read().strip())))
    with tqdm(os.listdir(os.path.join(datapath, "pos")), ncols=100) as t:
        for idx, filename in enumerate(t):
            if idx >= num:
                break
            fid, label = os.path.splitext(filename)[0].split("_")
            data.append("__label__{}\t{}".format(label, clean_data(open(os.path.join(datapath, "pos", filename)).read().strip())))
    if need_shuffle:
        shuffle(data)
    with open(foutpath, "w+") as fp:
        for s in data:
            fp.write(s + '\n')
    return

def create_vocab(foutpath):
    os.makedirs(foutpath, exist_ok=True)
    punctuation = [",","/",";","\"",":","?","!","@","#","$","%","^","&","(",")","_","+","=","~","<",">", "\\", "..."]
    unk = "<unk>"
    blank = "<blank>"
    seen = set([unk, blank] + punctuation)
    label = [1,2,3,4,7,8,9,10]
    with open(os.path.join(foutpath, "label.txt"), "w+") as fp:
        for tag in label:
            fp.write("{}\n".format(tag))
    with open(os.path.join(foutpath, "vocab.txt"), "w+") as fp:
        fp.write("{}\n".format(unk))
        fp.write("{}\n".format(blank))
        for p in punctuation:
            fp.write("{}\n".format(p))
        for line in open("data/aclImdb/imdb.vocab"):
            if line.strip() not in seen:
                seen.add(line.strip())
                fp.write("{}\n".format(line.strip()))
    return

def create_model_data(input_file, foutpath, filename, max_input_len=400):
    vocab = {line.strip():idx for idx, line in enumerate(open("data/tensorflow/vocab.txt"))}
    label = {line.strip():idx for idx, line in enumerate(open("data/tensorflow/label.txt"))}
    
    def padding(tokens):
        if len(tokens) >= max_input_len:
            return tokens[:max_input_len]
        else:
            return tokens + (max_input_len - len(tokens)) * [1]
    
    os.makedirs(foutpath, exist_ok=True)
    src = []
    tgt = []
    with tqdm(open(input_file), ncols=100) as t:
        for line in t:
            tmp = line.strip().split("\t")
            tag = label[tmp[0].lstrip("__label__")]
            tokens = padding([vocab.get(w, 0) for w in " ".join(tmp[1:]).split()])
            src.append(tokens)
            tgt.append(tag)
    pickle.dump(numpy.array(src), open(os.path.join(foutpath, "{}.src.pkl".format(filename)), "wb"))
    pickle.dump(numpy.array(tgt), open(os.path.join(foutpath, "{}.tgt.pkl".format(filename)), "wb"))
    return

if __name__ == "__main__":
    # create_fasttext_dataset("data/aclImdb/test", "data/fasttext/dev.txt", num=100, need_shuffle=False)
    # create_fasttext_dataset("data/aclImdb/test", "data/fasttext/test.txt", need_shuffle=False)
    # create_fasttext_dataset("data/aclImdb/train", "data/fasttext/train.txt")

    # create_vocab("data/tensorflow/")
    create_model_data("data/fasttext/dev.txt", "data/tensorflow/", "dev", max_input_len=128)
    create_model_data("data/fasttext/test.txt", "data/tensorflow/", "test", max_input_len=128)
    create_model_data("data/fasttext/train.txt", "data/tensorflow/", "train", max_input_len=128)