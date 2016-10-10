# TOHO_AI

東方projectに関する機械学習プロジェクト

1. [RNNLM](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)を使って言語モデルの作成。
それを使って[seq2seq](https://arxiv.org/pdf/1409.3215v3.pdf)的な何かで蓮子ちゃんチャットボット的な何かを作りたい

2. [能登氏](https://twitter.com/ntddk)の著書：「[深層強化学習による東方AI](https://booth.pm/ja/items/245254)」に感化されて、東方紺珠伝をクリアするAIを作りたい的な

# 目標
~~最終目標はオールクリア。そのためにもまずは1面クリアを目指す。~~

~~*結果などをまとめ、論文形式で冬コミに参加したい*~~

とりあえず①を冬コミ

# アプローチ

* RNNを用いた言語モデルの作成。RNNには[GRU](https://arxiv.org/pdf/1412.3555v1.pdf)を使う。Microsoftのりんなを参考に

* Deepleanring+強化学習で学習を行う。いわゆるGoogle DeepMindのDQNやAlphaGO