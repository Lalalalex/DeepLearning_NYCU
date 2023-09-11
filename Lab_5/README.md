# Lab 5
## Requirements
- Implement DQN
- Construct the neural network.
- Select action according to epsilon-greedy.
-  Construct Q-values and target Q-values.
- Calculate loss function.
- Update behavior and target network.
- Understand deep Q-learning mechanisms.
    - Implement DDPG
- Construct neural networks of both actor and critic.
- Select action according to the actor and the exploration noise.
- Update critic by minimizing the loss.
- Update actor using the sampled policy gradient.
- Update target network softly.
- Understand the mechanism of actor-critic.
## Datasets
- 三種遊戲，使用強化學習分別在種遊戲拿到高分
## Methods
- 這次作業基本上也是挖空自己填
- 比較特別的是breakout這個遊戲，它的state是當下畫面的圖片，但這個遊戲是連續的遊戲，你無法透過單一張照片知道接下來要幹嘛。因此必須將連續的state存起來當作模型的輸入。
- breakout要訓練比較久，大約要半天才會比較有起色。