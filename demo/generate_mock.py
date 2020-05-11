import pickle
import numpy as np


if __name__ == "__main__":
    lst = [
        {"video": "video0.mp4", "embeddings": np.random.rand(13, 100)},
        {"video": "video1.mp4", "embeddings": np.random.rand(1, 100)},
        {"video": "video2.mp4", "embeddings": np.random.rand(20, 100)},
    ]

    with open("mock.pkl", 'wb') as fout:
        pickle.dump(lst, fout, pickle.HIGHEST_PROTOCOL)

    with open("mock.pkl", 'rb') as fin:
        pickle.load(fin)