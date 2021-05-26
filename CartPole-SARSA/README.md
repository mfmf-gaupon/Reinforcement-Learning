# SARSAによるCartPole

TD制御アルゴリズムによるSARSAによりCartPole課題を解く  
環境はOpenAi gymから"CartPole-v1"を使用する

SARSAは方策オンのアルゴリズムである。

### 動作環境
- macOS Big Sur 11.2.3
- python 3.7.10
- gym 0.17.3

環境メモ  
open ai gymの0.18.0のバージョンだと、Monitorにて動画が上手く保存できない

### 使用法
```zsh:
% python sarsa.py
```

#### 参考コード
ベースはQ学習である
[シンプルな実装例で学ぶSARSA法およびモンテカルロ法](https://qiita.com/sugulu_Ogawa_ISID/items/7a14117bbd3d926eb1f2)
