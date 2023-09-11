# Lab 3
## Requirements
1. Implement the ResNet18, ResNet50, ResNet152 architecture on your own;
do not call the model from any library.
2. Train your model from scratch, do not load parameters from any
pretrained model.
3. Compare and visualize the accuracy trend between the 3 architectures, you
need to plot each epoch accuracy (not loss) during training phase and
testing phase.
4. Implement your own custom DataLoader
5. Design your own data preprocessing method
6. Calculate the confusion matrix and plotting 
## Datasets
- 白血球疾病辨識，主要是由白血球外觀判斷此病患是否有白血病(血癌)。
- 圖片檔和其對應label的CSV檔。
## Methods
- 這次的作業要實作ResNet18/50/152。三種架構差別不大，會寫一種就會寫其他種。
- 我把Block寫成一個class，一個Block裡面會包含複數卷基層。裡面會使用short cut技術，簡單來說就是把這個Block的input和output相加，得到殘差學習的效果。
    - 如果輸入輸出維度不相等就要downsample，經過一層卷積讓它們維度相等。

## Demo
- 1.跑一次看accuracy，解釋模型
- 2.解釋妳的train怎麼寫的
- 3.解釋為甚麼卷積可以接受不同長度的輸入。
    - 因為CNN本身的運算特性和adaptive pooling
- 4.解釋Overfitting
    - 模型過於複雜，過度擬和訓練資料的特徵，導致其不具有泛化性或是在測試資料表現不佳。
- 5.ResNet如何解決梯度消失
    - 加一個殘差網路，讓模型接收更往前的梯度，避免因為模型過深造成梯度消失
- 6.為何隨意壓縮圖片會導致模型結果不佳
    - 因為壓縮圖片會失去原先部分特徵，若損失的是重要特徵，可能會造成模型結果不佳