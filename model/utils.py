import json
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
        experiments = []
        for d in Path(exper_dir).iterdir():
            n = re.fullmatch(r'\d+', d.name)
            if d.is_dir() and n is not None:
                experiments.append(int(n.group()))
        experiments = sorted(experiments)        
        new_experiment = str(experiments[-1]+1)
    else:
        Path(exper_dir).mkdir(exist_ok=False)
        new_experiment = '0'
    new_exper_dir = Path(exper_dir).joinpath(new_experiment)
    Path(new_exper_dir).mkdir(exist_ok=False)
    return new_exper_dir

