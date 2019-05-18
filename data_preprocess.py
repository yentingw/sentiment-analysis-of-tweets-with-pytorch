import numpy as np
import re

from collections import Counter

# Remove tweet number, urls and hashtags etc
def clean_data(file):
  ls = file.split('\n')
  new_ls = []
  for i in ls:
    i = i.lower()
    j = re.sub(r'@[A-Za-z0-9]+','',i)
    k = re.sub('https?://[A-Za-z0-9./]+','',j)
    m = re.sub('[^a-zA-Z]', ' ', k)
    n = m.lstrip()
    new_ls.append(n)
  return new_ls

# Tokenize tweets
def convert_to_int(ls):
  words = []
  for tweet in ls:
    sentence = tweet.split()
    for word in sentence:
      words.append(word)
  
  counts = Counter(words)
  vocab = sorted(counts, key=counts.get, reverse=True)
  vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
  
  new_ls = []
  for i in ls:
    new_ls.append([vocab_to_int[word] for word in i.split()])
  new_ls = np.array(new_ls)
  return new_ls

# Tokenize labels
def label_to_int(ls):
  new_ls = []
  for i in ls:
    if i == 'positive':
      new_ls.append(1)
    elif i == 'neutral':
      new_ls.append(0)
    else:
      new_ls.append(-1)
  return np.array(new_ls)