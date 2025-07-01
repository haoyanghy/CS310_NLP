import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel
from parse_utils import DependencyArc, DependencyTree, State, parse_conll_relation
from get_train_data import FeatureExtractor

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="./models/wordpos_model.pt")
argparser.add_argument("--words_vocab", default="./vocab/words_vocab.txt")
argparser.add_argument("--pos_vocab", default="./vocab/pos_vocab.txt")
argparser.add_argument("--rel_vocab", default="./vocab/rel_vocab.txt")


class Parser(object):
    def __init__(self, extractor: FeatureExtractor, model_file: str):
        word_vocab_size = len(extractor.word_vocab)
        pos_vocab_size = len(extractor.pos_vocab)
        output_size = len(extractor.rel_vocab)
        if model_file.lower().endswith("base_model.pt"):
            self.model = BaseModel(word_vocab_size, output_size)
            self.model_type = "base_model.pt"
        else:
            self.model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
            self.model_type = "wordpos_model.pt"
        self.model.eval()
        self.model.eval()  # Set model to evaluation mode
        self.model.load_state_dict(torch.load(model_file, weights_only=True))
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.rel_vocab.items()]
        )

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            if self.model_type == "base_model.pt":
                input_repr = self.extractor.get_input_repr_word(words, pos, state)
            elif self.model_type == "wordpos_model.pt":
                input_repr = self.extractor.get_input_repr_wordpos(words, pos, state)
            else:
                print(
                    f"Unknown model type: {self.model_type}. Please use either 'base_model.pt' or 'wordpos_model.pt'."
                )
            input_tensor = torch.from_numpy(input_repr).unsqueeze(0).to(torch.long)
            prediction = self.model(input_tensor)
            best_action = self.output_labels[torch.argmax(prediction).item()]

            # Perform the action
            if best_action[0] == "shift" and state.buffer:
                state.shift()
            elif best_action[0] == "left_arc" and len(state.stack) > 1:
                state.left_arc(best_action[1])
            elif best_action[0] == "right_arc" and state.stack:
                state.right_arc(best_action[1])
            else:
                if state.buffer:
                    state.shift()

        # After processing the buffer, we need to pop all remaining items from the stack
        # and add them as dependencies to the root (0)
        while len(state.stack) > 1:
            word_id = state.stack.pop()
            state.deps.add((0, word_id, "dep"))

        res = DependencyTree()
        for parent, child, label in state.deps:
            deprel = DependencyArc(child, words[child], pos[child], parent, label)
            res.add_deprel(deprel)
        return res


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
    parser = Parser(extractor, args.model)

    # Test an example sentence, 3rd example from dev.conll
    words = [
        None,
        "The",
        "bill",
        "intends",
        "to",
        "restrict",
        "the",
        "RTC",
        "to",
        "Treasury",
        "borrowings",
        "only",
        ",",
        "unless",
        "the",
        "agency",
        "receives",
        "specific",
        "congressional",
        "authorization",
        ".",
    ]
    pos = [
        None,
        "DT",
        "NN",
        "VBZ",
        "TO",
        "VB",
        "DT",
        "NNP",
        "TO",
        "NNP",
        "NNS",
        "RB",
        ",",
        "IN",
        "DT",
        "NN",
        "VBZ",
        "JJ",
        "JJ",
        "NN",
        ".",
    ]

    tree = parser.parse_sentence(words, pos)
    print(tree)
