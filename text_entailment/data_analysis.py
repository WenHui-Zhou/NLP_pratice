import jsonlines
import pdb

train_file = './data/snli/snli_1.0/snli_1.0_train.jsonl'

train = []
label = []
label_map = {'contradiction':0, 'entailment':1, 'neutral':2}

with open(train_file,'r') as f:
    for item in jsonlines.Reader(f):
        train.append([item['sentence1'],item['sentence2']])
        label.append(label_map[item['gold_label']])

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
from tqdm

from keras.utils import to_categorical
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split # 划分训练集和测试集

set_random_seed(123)
random.seed(123)

def clean_sentences(df):
    reviews = []
    for sent in tqdm(df):







