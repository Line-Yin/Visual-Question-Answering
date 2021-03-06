import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class Enc(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Enc, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Dec(nn.Module):
    def __init__(self, embed_size, hidden_size, ques_vocab_size, ans_vocab_size, num_layers, max_seq_length=26):
        """Set the hyper-parameters and build the layers."""
        super(Dec, self).__init__()
        self.embed = nn.Embedding(ques_vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, ans_vocab_size)
        # self.classifier = Classifier(hidden_size * 2, 256, ans_vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        # print(embeddings.shape)
        # print(features.unsqueeze(1).shape)
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        hiddens, hn = self.rnn(packed)

        # tmp = torch.nn.utils.rnn.pad_packed_sequence(hn[-1], batch_first=True)

        # hiddens, hidden_lengths = tmp

        # print(features.shape)
        # print(hiddens[:, -1, :].shape)

        # print(hn.shape)
        # print(hn[-1].shape)
        # print(features.shape)

        hiddens = torch.cat((features, hn[-1]), 1)

        # print(hiddens.shape)

        outputs = self.linear(hiddens)
        # outputs = self.classifier(hiddens)
        outputs = F.log_softmax(outputs, dim=1)

        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.2):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))
