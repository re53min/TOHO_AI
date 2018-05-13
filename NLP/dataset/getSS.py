# -*- coding: utf-8 -*-

import codecs
import io
import json
import sys

from urllib import request

import requests
from bs4 import BeautifulSoup


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_ss_info(dic):
    """
    ssさがすよAPIをたたいてキーワード検索した
    結果を返す。
    その後、jsonの書き出しを行う。
    :param dic: config.json
    :return: ssインフォ
    """

    # ss探すよAPIを叩く
    url = dic['url']+"tag={}&num={}&ord={}"
    info = requests.get(url.format(dic["tag"], dic["num"], dic["ord"])).json()

    # jsonデータの出力
    with codecs.open('ss_info.json', 'w', 'utf-8') as ss_info:
        json.dump(info, ss_info, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    return info


def get_ss(file):
    """
    東方創想話からssをスクレイピング
    :return:
    """

    ss_body = ''
    for ss in get_ss_info(json.load(file)):
        try:
            with request.urlopen(ss['link']) as res:  # URLオープン
                soup = BeautifulSoup(res.read(), 'lxml')
                ss_text = soup.find('div', {'id': 'contentBody'}).text  # ss本文の取得
                print(ss_text.replace('。', '。\n'))
                # ss_body += ss_text
                # ssの本文データを保存
                # with codecs.open('ss\\'+ss['title']+'.txt', 'w', 'utf-8') as raw_data:
                # raw_data.write(ss_text)
        except AttributeError:
            continue
        except OSError:
            continue

    return ss_body


if __name__ == "__main__":

    with codecs.open("config.json", "r", 'utf-8') as f:
        get_ss(f)
        # print(get_ss(f))
