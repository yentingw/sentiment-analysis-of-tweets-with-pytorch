import re
import numpy as np
import torch
import torch.nn as nn

from collections import Counter
from data_preprocess import clean_data, convert_to_int, label_to_int
from rnn import SentimentRNN

with open('data/train-tweets.txt', 'r') as f:
    train_tweets = f.read()
with open('data/train-labels.txt', 'r') as f:
    train_labels = f.read()
with open('data/eval-tweets.txt', 'r') as f:
    eval_tweets = f.read()
with open('data/eval-labels.txt', 'r') as f:
    eval_labels = f.read()
with open('data/test-tweets.txt', 'r') as f:
    test_tweets = f.read()

train_tweets_ls = clean_data(train_tweets)
train_labels_ls = clean_data(train_labels)
eval_tweets_ls = clean_data(eval_tweets)
eval_labels_ls = clean_data(eval_labels)
test_tweets_ls = clean_data(test_tweets)

train_ints = convert_to_int(train_tweets_ls)
eval_ints = convert_to_int(eval_tweets_ls)
test_ints = convert_to_int(test_tweets_ls)

train_labels_ints = label_to_int(train_labels_ls)
eval_labels_ints = label_to_int(eval_labels_ls)

# Remove empty tweets
train_tweets_lens = Counter([len(x) for x in train_ints])
print("Zero-length train tweets: {}".format(train_tweets_lens[0]))
print("Maximum train tweets length: {}".format(max(train_tweets_lens)))

eval_tweets_lens = Counter([len(x) for x in eval_ints])
print("Zero-length eval tweets: {}".format(eval_tweets_lens[0]))
print("Maximum eval tweets length: {}".format(max(eval_tweets_lens)))

test_tweets_lens = Counter([len(x) for x in test_ints])
print("Zero-length test tweets: {}".format(test_tweets_lens[0]))
print("Maximum test tweets length: {}".format(max(test_tweets_lens)))

print('Number of train tweets before removing outliers: ', len(train_ints))
non_zero_idx = [ii for ii, tweet in enumerate(train_ints) if len(tweet) != 0]
train_ints = [train_ints[ii] for ii in non_zero_idx]
train_labels_ints = np.array([train_labels_ints[ii] for ii in non_zero_idx])
print('Number of train tweets after removing outliers: ', len(train_ints))

print('Number of eval tweets before removing outliers: ', len(eval_ints))
non_zero_idx = [ii for ii, tweet in enumerate(eval_ints) if len(tweet) != 0]
eval_ints = [eval_ints[ii] for ii in non_zero_idx]
eval_labels_ints = np.array([eval_labels_ints[ii] for ii in non_zero_idx])
print('Number of eval tweets after removing outliers: ', len(eval_ints))

print('Number of test tweets before removing outliers: ', len(test_ints))
non_zero_idx = [ii for ii, tweet in enumerate(test_ints) if len(tweet) != 0]
test_ints = [test_ints[ii] for ii in non_zero_idx]
print('Number of test tweets after removing outliers: ', len(test_ints))


def pad_features(ls_ints, seq_length):
    features = np.zeros((len(ls_ints), seq_length), dtype=int)
    for i, row in enumerate(ls_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

seq_length = 20

train_features = pad_features(train_ints, seq_length=seq_length)
eval_features = pad_features(eval_ints, seq_length=seq_length)
test_features = pad_features(test_ints, seq_length=seq_length)

assert len(train_features)==len(train_ints), "Your features should have as many rows as reviews."
assert len(train_features[0])==seq_length, "Each feature row should contain seq_length values."

from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(torch.from_numpy(train_ints), torch.from_numpy(train_labels_ints))
eval_data = TensorDataset(torch.from_numpy(eval_ints), torch.from_numpy(eval_labels_ints))

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
eval_loader = DataLoader(eval_data, shuffle=True, batch_size=batch_size)

vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 128
n_layers = 1

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 3
counter = 0
print_every = 100
clip = 5

net.train()

for e in range(epochs):
    
    h = net.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
    
        h = tuple([each.data for each in h])

        net.zero_grad()

        output, h = net(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in eval_loader:
                val_h = tuple([each.data for each in val_h])
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))