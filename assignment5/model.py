import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        # Input: 6 word IDs (from get_input_repr_word)
        embedding_dim = 50
        hidden_dim = 400
        dropout_rate = 0.3

        # Embedding layer for words
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim)

        # Feedforward network
        self.fc1 = nn.Linear(6 * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # x: (batch_size, 6) - 6 word IDs (stack top 3 + buffer top 3)
        batch_size = x.size(0)

        # Get embeddings: (batch_size, 6, embedding_dim)
        x = self.word_embedding(x)

        # Flatten: (batch_size, 6 * embedding_dim)
        x = x.view(batch_size, -1)

        # Feedforward
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)  # Output log probabilities
        ### END YOUR CODE ###
        return x


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        ### START YOUR CODE ###
        # Input: 12 features (6 word IDs + 6 POS IDs from get_input_repr_wordpos)
        word_embedding_dim = 100
        pos_embedding_dim = 50
        hidden_dim = 400
        dropout_rate = 0.3

        # Embedding layers
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)

        # Feedforward network
        input_dim = 6 * (word_embedding_dim + pos_embedding_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # x: (batch_size, 12) - first 6 are word IDs, last 6 are POS IDs
        batch_size = x.size(0)

        # Split input
        word_ids = x[:, :6]  # (batch_size, 6)
        pos_ids = x[:, 6:]  # (batch_size, 6)

        # Get embeddings
        word_emb = self.word_embedding(word_ids)  # (batch_size, 6, word_embedding_dim)
        pos_emb = self.pos_embedding(pos_ids)  # (batch_size, 6, pos_embedding_dim)

        # Concatenate word and POS embeddings: (batch_size, 6, word_embedding_dim + pos_embedding_dim)
        x = torch.cat((word_emb, pos_emb), dim=-1)

        # Flatten: (batch_size, 6 * (word_embedding_dim + pos_embedding_dim))
        x = x.view(batch_size, -1)

        # Feedforward
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)  # Output log probabilities
        ### END YOUR CODE ###
        return x
