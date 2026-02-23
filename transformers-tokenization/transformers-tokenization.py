import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        # "PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3 => len = 4
        self.word_to_id = {word: id for id, word in enumerate(special_tokens)}

        all_tokens = " ".join(texts).split()
        unique_tokens = sorted(set(all_tokens))

        current_id = len(self.word_to_id) #4
        for token in unique_tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = current_id
                current_id += 1
        
        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        unk_id = self.word_to_id[self.unk_token]
        ids = []
        for word in text.split():
            ids.append(self.word_to_id.get(word, unk_id))
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        tokens = [ self.id_to_word.get(id, self.unk_token) for id in ids]
        words = " ".join(tokens)
        return words
        
