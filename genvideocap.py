import os
import sys
import pandas as pd
import numpy as np

video_dir = './data/webvid/data/videos'

df = pd.read_csv('./data/webvid/results_2M_train.csv')

df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])), axis=1)
df['rel_fn'] = df['rel_fn'] + '.mp4'

# remove nan
df.dropna(subset=['page_dir'], inplace=True)

playlists_to_dl = np.sort(df['page_dir'].unique())

print("Generate train.txt...")

f = open("./data/webvid/train.txt", "w")

for page_dir in playlists_to_dl:
    vid_dir_t = os.path.join(video_dir, page_dir)
    pdf = df[df['page_dir'] == page_dir]
    if len(pdf) > 0:
        for idx, row in pdf.iterrows():
            video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
            if os.path.exists(video_fp):
                print(video_fp[14:], row['name'].replace("\n","").replace("\r",""), file=f)

f.close()
