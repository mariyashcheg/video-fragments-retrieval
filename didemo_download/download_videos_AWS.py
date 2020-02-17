''' 
Code to get the data in the DiDeMo video dataset using the videos stored on AWS.

Usage:

python download_videos_AWS.py  --download --video_directory DIRECTORY

will download videos from flickr to DIRECTORY

python download_videos_AWS.py  
'''
import sys
import urllib3
import argparse
import os
import json
import warnings

# sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument("--video_directory", type=str, default='videos/', help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", dest="download", action="store_true")
parser.set_defaults(download=False)

args = parser.parse_args()

DATA_PATH = '/data/%s' % args.video_directory
if args.download:
    print(DATA_PATH)
    assert os.path.exists(DATA_PATH)

DATA_TEMPLATE = 'data/%s_data.json'
MULTIMEDIA_TEMPLATE = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4'

def get_aws_link(h):
    return MULTIMEDIA_TEMPLATE % (h[:3], h[3:6], h)

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def read_hash(hash_file):
    lines = open(hash_file).readlines()
    yfcc100m_hash = {}
    for line_count, line in enumerate(lines):
         sys.stdout.write('\r%d/%d' %(line_count, len(lines)))
         line = line.strip().split('\t')
         yfcc100m_hash[line[0]] = line[1]
    print("\n")
    return yfcc100m_hash

splits = ['test', 'val', 'train']
caps = [] 
for split in splits:
     caps.extend(read_json(DATA_TEMPLATE % split))
videos = set([cap['video'] for cap in caps])

yfcc100m_hash = read_hash('data/yfcc100m_hash.txt')

missing_videos = []
url_manager = urllib3.PoolManager()

warnings.filterwarnings("ignore")
for video_count, video in enumerate(videos):
    sys.stdout.write('\rDownloading video: %d/%d\n' % (video_count + 1, len(videos)))
    video_id = video.split('_')[1]
    link = get_aws_link(yfcc100m_hash[video_id])
    if args.download:
        try:
            response = url_manager.request('GET', link)
            with open('%s/%s.mp4' % (DATA_PATH, video), 'wb') as f:
                f.write(response.data)
        except:
            missing_videos.append(video)
            print("Could not download link: %s\n" % link)
    else:
        try:
            response = url_manager.request('GET', link)
        except:
            missing_videos.append(video)
            print("Could not find link: %s\n".format(link))

if len(missing_videos) > 0:
    write_txt = open('missing_videos.txt', 'w')
    for video in missing_videos:
        write_txt.writelines('%s\n' %video)
    write_txt.close()
