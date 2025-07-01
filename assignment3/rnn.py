import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import numpy as np


def download_nltk_resources():
    nltk.data.path.append("/home/tangm_lab/cse12212027/NLP/nltk_data")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        raise RuntimeError(
            "NLTK 'punkt' data not found. Please manually install as described above."
        )


# def download_nltk_resources():
#     try:
#         nltk.data.find("tokenizers/punkt_tab")
#     except LookupError:
#         nltk.download("punkt_tab")
#     try:
#         nltk.data.find("tokenizers/punkt")
#     except LookupError:
#         nltk.download("punkt")


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()  # Convert to lowercase
    return text


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def build_vocab(tokens, max_vocab_size=20000):
    word_counts = Counter(tokens)
    vocab = {
        word: idx + 2
        for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size - 2))
    }
    # vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}
    vocab["<PAD>"] = 0  # Padding token
    vocab["<UNK>"] = 1  # Unknown token for out-of-vocab words
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word


def tokens_to_tensor(tokens, vocab):
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)


class HarryPotterDataset(Dataset):
    def __init__(self, tensor, sequence_length):
        self.tensor = tensor
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tensor) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = self.tensor[idx : idx + self.sequence_length]
        target_seq = self.tensor[idx + 1 : idx + self.sequence_length + 1]
        return input_seq, target_seq


def preprocess_and_load(
    file_path, sequence_length=20, batch_size=128, max_vocab_size=20000
):
    download_nltk_resources()
    text = load_text(file_path)
    tokens = tokenize_text(text)
    vocab, idx_to_word = build_vocab(tokens, max_vocab_size)
    tensor = tokens_to_tensor(tokens, vocab)
    dataset = HarryPotterDataset(tensor, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader, vocab, idx_to_word, len(tokens)


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        return hidden


def train_model(model, train_loader, val_loader, num_epochs, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (input_seq, target_seq) in enumerate(train_loader):
            if i % 1000 == 0:
                print(f"Batch {i}/{len(train_loader)}")
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            hidden = model.init_hidden(input_seq.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)

            optimizer.zero_grad()
            output, hidden = model(input_seq, hidden)
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                hidden = model.init_hidden(input_seq.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)
                output, hidden = model(input_seq, hidden)
                val_loss += criterion(
                    output.view(-1, vocab_size), target_seq.view(-1)
                ).item()
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
    return model


def compute_perplexity(model, test_loader, device):
    model.eval()
    loss_fn = nn.NLLLoss(ignore_index=0, reduction="none")
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for input_seq, target_seq in test_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            hidden = model.init_hidden(input_seq.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)

            output, hidden = model(input_seq, hidden)
            log_probs = torch.log_softmax(output, dim=-1)
            loss = loss_fn(log_probs.view(-1, vocab_size), target_seq.view(-1))
            total_loss += loss.sum().item()
            total_tokens += (target_seq != 0).sum().item()  # Count non-padding tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


# Greedy search generation
def generate_sentence(
    model, start_tokens, vocab, idx_to_word, max_length=20, device="cuda"
):
    model.eval()
    input_seq = (
        torch.tensor(
            [vocab.get(token, vocab["<UNK>"]) for token in start_tokens],
            dtype=torch.long,
        )
        .unsqueeze(0)
        .to(device)
    )
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)

    generated = start_tokens.copy()
    with torch.no_grad():
        for _ in range(max_length - len(start_tokens)):
            output, hidden = model(input_seq, hidden)
            next_token_idx = torch.argmax(output[:, -1, :], dim=-1).item()
            generated.append(idx_to_word[next_token_idx])
            input_seq = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)

    return " ".join(generated)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = "Harry_Potter_all_books_preprocessed.txt"
    sequence_length = 20
    batch_size = 128
    vocab_size = 20000

    download_nltk_resources()
    text = load_text(file_path)
    tokens = tokenize_text(text)
    vocab, idx_to_word = build_vocab(tokens, vocab_size)
    tensor = tokens_to_tensor(tokens, vocab)
    dataset = HarryPotterDataset(tensor, sequence_length)

    # 90%-5%-5% split
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model hyperparameters
    embedding_dim = 200
    hidden_dim = 128
    num_layers = 2
    dropout = 0.2
    num_epochs = 10

    rnn_model = RNNLanguageModel(
        vocab_size, embedding_dim, hidden_dim, num_layers, dropout
    )

    # Train models
    print("Training RNN...")
    rnn_model = train_model(rnn_model, train_loader, val_loader, num_epochs, device)

    # Compute perplexity
    rnn_perplexity = compute_perplexity(rnn_model, test_loader, device)
    print(f"\nRNN Test Perplexity: {rnn_perplexity:.2f}")

    # Generate sentences
    prefixes = [
        ["harry", "looked"],
        ["the", "wand"],
        ["hermione", "said"],
        ["ron", "grabbed"],
        ["dumbledore", "smiled"],
    ]

    print("\nGenerated Sentences:")
    for prefix in prefixes:
        rnn_sentence = generate_sentence(
            rnn_model, prefix, vocab, idx_to_word, device=device
        )
        print(f"Prefix: {' '.join(prefix)}")
        print(f"RNN: {rnn_sentence}")
