import os
import sys
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from get_train_data import FeatureExtractor
from model import BaseModel, WordPOSModel

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_file", default="./extract/input_train.npy")
argparser.add_argument("--target_file", default="./extract/target_train.npy")
# argparser.add_argument("--input_file", default="./extract/input_train_wordpos.npy")
# argparser.add_argument("--target_file", default="./extract/target_train_wordpos.npy")
argparser.add_argument("--words_vocab", default="./vocab/words_vocab.txt")
argparser.add_argument("--pos_vocab", default="./vocab/pos_vocab.txt")
argparser.add_argument("--rel_vocab", default="./vocab/rel_vocab.txt")
argparser.add_argument(
    "--model",
    default="wordpos_model.pt",
    help="path to save model file, if not specified, a .pt with timestamp will be used",
)


if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(
            f"Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}"
        )
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.rel_vocab)

    ### START YOUR CODE ###
    # TODO: Initialize the model
    if args.model and "base_model.pt" in args.model.lower():
        model = BaseModel(word_vocab_size, output_size)
    else:
        model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    ### END YOUR CODE ###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5
    )

    inputs = np.load(args.input_file)
    targets = np.load(args.target_file)  # pytorch input is int
    print("Done loading data.")

    # Train loop
    n_epochs = 10
    print_loss_every = 10000  # every 10000 batches
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    ### START YOUR CODE ###
    # TODO: Wrap inputs and targets into tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    ### END YOUR CODE ###

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    val_split = 0.1  # 10% for validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    n_batches = len(train_loader)

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:

            ### START YOUR CODE ###
            # TODO: Get inputs and targets from batch; feed inputs to model and compute loss; backpropagate and update model parameters
            batch_inputs, batch_targets = batch
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(batch_inputs)  # Forward pass
            loss = criterion(outputs, batch_targets)  # Compute loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update parameters
            ### END YOUR CODE ###

            epoch_loss += loss.item()
            batch_count += 1
            if batch_count % print_loss_every == 0:
                avg_loss = epoch_loss / batch_count
                sys.stdout.write(
                    f"\rEpoch {epoch+1}/{n_epochs} - Batch {batch_count}/{n_batches} - Loss: {avg_loss:.4f}\n"
                )
                sys.stdout.flush()

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_inputs, batch_targets = batch
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                val_count += 1

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)  # Step scheduler with validation loss

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            os.makedirs("models", exist_ok=True)
            model_filename = (
                args.model
                if args.model
                else f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            model_path = os.path.join("models", model_filename)
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        epoch_end_time = time.time()
        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f} sec"
        )

    print(f"Model saved to {model_path}")
