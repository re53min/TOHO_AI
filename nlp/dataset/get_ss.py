# -*- coding: utf-8 -*-

import codecs
import io
import json
import sys
import re

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


def clean_text(text):

    replace_text = re.sub(r'　', '', text)
    replace_text = re.sub('\u3000', '', replace_text)
    replace_text = re.sub(r'[「」]', '', replace_text)

    return replace_text


def get_ss(file):
    """
    東方創想話からssをスクレイピング
    :return:
    """

    ss_body = {}
    for ss_n, ss in enumerate(get_ss_info(json.load(file))):
        try:
            with request.urlopen(ss['link']) as res:  # URLオープン
                soup = BeautifulSoup(res.read(), 'lxml')
                for br in soup.find_all("br"):
                    br.replace_with("\n")
                ss_text = soup.find('div', {'id': 'contentBody'}).text  # ss本文の取得
                ss_text = re.findall(r"「.*」", ss_text)

                for n, tmp in enumerate(ss_text):
                    ss_body[n] = clean_text(tmp)
                print(ss_body)

                # ssの本文データを保存
                with codecs.open('ss\\{id}_{title}.json'.format(id=ss_n, title=ss['title']), 'w', 'utf-8') as ss_json:
                    json.dump(ss_body, ss_json, ensure_ascii=False, indent=4)

        except AttributeError:
            continue
        except OSError:
            continue

    return ss_body


if __name__ == "__main__":

    with codecs.open("config.json", "r", 'utf-8') as f:
        get_ss(f)
        # print(get_ss(f))
