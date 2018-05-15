# -*- coding:utf-8 -*-

import json, pickle
from pathlib import Path

"""
http://www.anlp.jp/proceedings/annual_meeting/2015/html/paper/WS_PNN23_dial.pdf
"""


p = Path("./json/rest1046/")
speaker = []
response = []
prev = None

for f in list(p.glob("*.json")):
    with open(f, 'r', encoding='utf-8') as data_file:
        json_data = json.load(data_file)

        for turn in json_data["turns"]:
            # 無言処理
            if turn["utterance"] == "":
                turn["utterance"] = "無言"

            if prev == turn["speaker"]:
                continue
            elif turn["speaker"] == "S":
                speaker.append(turn["utterance"])
                s = turn["speaker"] + ":" + turn["utterance"]
                print(s)
            elif turn["speaker"] == "U":
                response.append(turn["utterance"])
                s = turn["speaker"] + ":" + turn["utterance"]
                print(s)
            prev = turn["speaker"]
            # s = turn["speaker"] + ":" + turn["utterance"]
            # print(s)

print(len(speaker))
print(len(response))

with open('./json/speaker.pickle', 'wb') as f:
    pickle.dump(speaker, f)

with open('./json/response.pickle', 'wb') as f:
    pickle.dump(response, f)


with open('./json/speaker.pickle', 'rb') as f:
    x = pickle.load(f)

with open('./json/response.pickle', 'rb') as f:
    y = pickle.load(f)


for i, tmp in enumerate(x):
    if tmp == "":
        print("{}: {}".format(i, tmp))
