# -*- coding:utf-8 -*-

import json
import pickle
from pathlib import Path

"""
http://www.anlp.jp/proceedings/annual_meeting/2015/html/paper/WS_PNN23_dial.pdf
"""


p = Path("./ss/")
merry = []
renko = []
prev = None

for name in list(p.glob("*.json")):
    with open(name, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
            [merry.append(tmp) for tmp in list(json_data.values())[0::2]]
            [renko.append(tmp) for tmp in list(json_data.values())[1::2]]
        except Exception as e:
            print(e)
            pass

print(len(merry))
print(len(renko))

with open('./ss/merry.pickle', 'wb') as f:
    pickle.dump(merry, f)

with open('./ss/renko.pickle', 'wb') as f:
    pickle.dump(renko, f)


with open('./ss/merry.pickle', 'rb') as f:
    x = pickle.load(f)

with open('./ss/renko.pickle', 'rb') as f:
    y = pickle.load(f)

for tmp_x, tmp_y in zip(x, y):
    print("メリー: {}".format(tmp_x))
    print("蓮子: {}".format(tmp_y))
