from collections import defaultdict
import numpy as np
import pickle
import torch


def load_model(path):
    return torch.load(path)


class Worker:
    def __init__(self, index_path, textual_model_path):
        with open(index_path, 'rb') as fin:
            index = pickle.load(fin)
            # suppose index is a list of dicts:
            # { video: video_path, embeddings: nd.array with shape [fragments, dimsize]
            # let's make dummy indexer:
            vectors = []
            idx2fragment = dict()
            for entry in index:
                v = entry["embeddings"]
                vectors.append(v)  # number of fragments
                for i in range(len(v)):
                    idx2fragment[i] = (entry["video"], i)

        self.index_vectors = np.concatenate(vectors)
        self.idx2fragment = idx2fragment
        self.textual_model = torch.jit.load(textual_model_path)

    def process(self, query):
        # todo: add query encoding
        embedded_query = self.textual_model(query)  # suppose [1, dim] shape
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
                    video=video,
                    fragments=fragments,
                )
            )
        return ret
