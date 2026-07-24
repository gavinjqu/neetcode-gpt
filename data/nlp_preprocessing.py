import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        sentence = positive + negative
        all_words = set()
        for sent in sentence:
            all_words.update(sent.split()) #now they are split into strings
        vocab = sorted(all_words)
        word_id = {word: i + 1 for i, word in enumerate(vocab)} # comprehension on dict
        
        #step 2 encode each sentence as float:
        encoding = [[word_id[word] for word in sent.split()] for sent in sentence]

        # step 3: convert to tensor
        tensors = [torch.tensor(ids) for ids in encoding]

        # step 4: 
        result = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
        return result