from collections import defaultdict
import pickle
import re
import string
import sys

import numpy as np
import torch

sys.path.insert(0, "../model")
import data


def load_model(path):
    return torch.load(path)


class Worker:
    def __init__(self, index_path, textual_model_path, glove_path, max_query_len=50):
        print(index_path)
        with open(index_path, 'rb') as fin:
            index = pickle.load(fin)
            # suppose index is a list of dicts:
            # { video: video_path, embeddings: nd.array with shape [fragments, dimsize]
            # let's make dummy indexer:
            vectors = []
            idx2fragment = dict()
            counter = 0
            for entry in index:
                v = entry["embeddings"]
                vectors.append(v)  # number of fragments
                for i in range(len(v)):
                    idx2fragment[counter] = (entry["video"], i)
                    counter += 1

        self.word_indexer = data.WordIndexer(glove_path)
        self.max_query_len = max_query_len

        self.index_vectors = np.concatenate(vectors)
        self.idx2fragment = idx2fragment
        self.textual_model = torch.jit.load(textual_model_path)

    def process(self, query):
        query = query.rstrip('\n ').lower()
        words = re.sub(f'(?=[^\s])(?=[{string.punctuation}])', ' ', query).split(' ')
        encoded = self.word_indexer.items2tensor([words], self.max_query_len)

        with torch.no_grad():
            embedded_query = self.textual_model(encoded).numpy()  # suppose [1, dim] shape
        distances = np.linalg.norm(self.index_vectors - embedded_query, axis=1)
        idx = np.argsort(distances)
        top5 = defaultdict(list)
        for i in idx:
            video, fragment = self.idx2fragment[i]
            top5[video].append(fragment)
            if len(top5) >= 5:
                break

        ret = []
        for video, fragments in top5.items():
            ret.append(
                dict(
                    video="/videos/" + video,  # just a mapping from local to url
                    fragments=fragments,
                )
            )
        return ret
