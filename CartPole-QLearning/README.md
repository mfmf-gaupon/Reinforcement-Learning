# Q学習によるCartPole

TD制御アルゴリズムによるQ学習によりCartPole課題を解く  
環境はOpen AI Gymから"CartPole-v1"を使用する

なおこのコードは完全方策オフでなく部分的な方策オフである。
(行動を選ぶのにεグリーディ法を使っているため)

### 動作環境
- macOS Big Sur 11.2.3
- python 3.7.10
- gym 0.17.3

環境メモ  
open ai gymの0.18.0のバージョンだと、Monitorにて動画が上手く保存できない

### 使用法
```zsh:
% python ql.py
```

#### 参考コード
『TensorFlowによる深層強化学習入門』オーム社
