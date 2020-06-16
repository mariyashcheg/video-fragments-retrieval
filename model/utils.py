import json
import numpy as np
import itertools
import re
from pathlib import Path

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


def load_missed_videos(missed_videos_path):
    # no features for these videos
    missed_videos_path = Path(missed_videos_path).joinpath('missed_videos_features.json')
    if missed_videos_path.exists():
        with open(missed_videos_path, 'r') as f:
            missed_videos = json.load(f)
    else:
        missed_videos = []
    return missed_videos


def load_dataset_info(dataset_type, dataset_directory, missed_videos_path):
    annotations = {}
    videos = []
    missed_videos = load_missed_videos(missed_videos_path)

    for video in read_json(Path(dataset_directory).joinpath(f'{dataset_type}_data.json')):
        if video['video'] not in missed_videos:
            annotations[video['annotation_id']] = dict(
                video=video['video'],
                description=video['description'],
                times=video['times']
            )
            videos.append(video['video'])
            
    return annotations, list(set(videos))


def start_new_experiment(exper_dir):
    if Path(exper_dir).exists() and Path(exper_dir).is_dir():
        experiments = get_existing_experiments(exper_dir)
        new_experiment = str(experiments[-1]+1)
    else:
        Path(exper_dir).mkdir(exist_ok=False)
        new_experiment = '0'
    new_exper_dir = Path(exper_dir).joinpath(new_experiment)
    Path(new_exper_dir).mkdir(exist_ok=False)
    return new_exper_dir


def get_existing_experiments(exper_dir):
    experiments = []
    for d in Path(exper_dir).iterdir():
        n = re.fullmatch(r'\d+', d.name)
        if d.is_dir() and n is not None:
            experiments.append(int(n.group()))
    return sorted(experiments)


def str2bool(param):
    if str(param).lower() == 'true':
        return True
    elif str(param).lower() == 'false':
        return False
    else:
        return None


def generate_moments(num_segments):
    moments = [(j,j) for j in range(num_segments)]
    for j in itertools.combinations(range(num_segments), 2):
        moments.append(j)
    # moments = [(i,j) for (i,j) in moments if j - i + 1 >= 2]
    return moments


def get_iou(times, start_t, end_t):
    times = np.array(times)
    intersection = np.maximum(np.minimum(times[:, 1], end_t) + 1 - np.maximum(times[:, 0], start_t), 0)
    union = np.maximum(times[:, 1], end_t) + 1 - np.minimum(times[:, 0], start_t)
    return intersection / union


def grad_norm(model):
    grad = 0.0
    count = 0
    for name, tensor in model.named_parameters():
        if tensor.grad is not None:
            grad += tensor.grad.data.norm().cpu().item()
            count += 1
    return grad / count