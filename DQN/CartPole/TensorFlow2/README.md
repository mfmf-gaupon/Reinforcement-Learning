# DQN(Deep-Q-Network)によるCartPole

TenseorFlow2(とKeras)でDQNを実装しCartPole課題を解く    
環境はOpen AI Gymから"CartPole-v1"を使用する

### 動作環境
- Windows10
- python 3.6.10
- gym 0.17.1
- tensorflow 2.4.1

## ハイパーパラメータとか
### バックボーン
- 2層MLP
    - 隠れ層のユニット数 : 100
    - 活性化関数 : tanh
- optimizer : Adam
    - learning rate : 4e-4

### ハイパーパラメータ
- バッチサイズ : 16
- gamma : 0.96
- update network period : 2
- update target period : 500

#### 参考コード,書籍など
[DQNの進化史 ①DeepMindのDQN](https://horomary.hatenablog.com/entry/2021/01/26/233351)
