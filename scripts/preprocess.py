import os
import re
import nltk
from collections import Counter
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# download dataset from HuggingFace
dataset = load_dataset("r-three/ag_news_subset")
train_data = [x["description"] for x in dataset["train"]]
valid_data = [x["description"] for x in dataset["validation"]]
test_data  = [x["description"] for x in dataset["test"]]

# clean text
def clean_text(text):
    # remove html tags
    text = re.sub(r'<[^>]+>', '', text)
    # remove urls
    text = re.sub(r'http\S+', '', text)
    text = text.replace(u'\ufeff', '')
    # remove extra spaces
    text = " ".join(text.split())
    # lowercase
    text = text.lower()
    return text

train_data = [clean_text(t) for t in train_data]
valid_data = [clean_text(t) for t in valid_data]
test_data  = [clean_text(t) for t in test_data]

# tokenize sentences
def split_sentences(data):
    result = []
    for text in data:
        sentences = sent_tokenize(text)
        # remove short sentences
        for s in sentences:
            if len(s.split()) > 3:
                result.extend([s.strip()])
    return result

train_data = split_sentences(train_data)
valid_data = split_sentences(valid_data)
test_data  = split_sentences(test_data)

# create a vocabulary and replace lower frequency words
vocab_size = 5000
all_tokens = " ".join(train_data).split()
counter = Counter(all_tokens)

vocab = set([w for w, _ in counter.most_common(vocab_size)])

def replace_unk(sentence):
    return " ".join([w if w in vocab else "<unk>" for w in sentence.split()])

train_data = [replace_unk(s) for s in train_data]
valid_data = [replace_unk(s) for s in valid_data]
test_data  = [replace_unk(s) for s in test_data]

# save texts
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

def save(data_list, path):
    text = " <news> ".join(data_list)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

save(train_data, os.path.join(output_dir, "train.txt"))
save(valid_data, os.path.join(output_dir, "valid.txt"))
save(test_data, os.path.join(output_dir, "test.txt"))

print(f"Train/valid/test texts have been saved.")