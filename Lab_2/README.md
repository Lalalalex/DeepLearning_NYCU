# Lab 2
## Requirements
1. Implement the EEGNet, DeepConvNet with three kinds of activation
function including ReLU, Leaky ReLU, ELU
2. In the experiment results, you have to show the highest accuracy (not loss) of
two architectures with three kinds of activation functions.
3. To visualize the accuracy trend, you need to plot each epoch accuracy (not
loss) during training phase and testing phase.
## Datasets
- 此次Dataset主要分為兩個檔案，S4b和X11b。
兩個檔案的大小分別都是540x2x750，然後根據助教提供的dataloader，會將這兩個檔案concat在一起，然後擴增一個維度。最後會變成1080x1x2x750。
- 各維度代表的意義：
    - 1080：這裡代表的是資料筆數，也就是總共有1080筆資料
    - 1：這是擴增之後多出來的維度，之所以要擴增資料維度是因為我們的每筆資料都是二維的(原先三維扣除掉第一維是資料筆數)。而pytorch的CNN預設是對三維資料進行運算，因此必須擴增這一維度。這個維度相當於原本CNN中的channel(e.g.RGB)
    - 2：這個是資料的channel，這裡的channel跟上面的channel是不一樣的，剛剛上面提到的是一般捲積運算中圖片資料的channel，這裡指的是我們電波圖的兩個channel。如果這部分要對應到CNN的運算，那相當於圖片的寬
    - 750：電波圖的時間軸，對應到CNN中的長
    - 綜上所述，我們這次的資料其實是總共有1080筆的電波圖，每個電波圖都有兩個channel並時間長750，最後再擴增出一維輔助計算
## Methods
- 這次的作業比較簡單，基本上就是根據助教給的模型架構接起來而已。
- 我這次額外做了一些簡單的Augmentation，有興趣可以看我的report。有幾個Autmentation對accuracy有不錯的幫助，上不去的人可以參考。
- 這次加分是設計一個模型，我自己是設計了一個簡易的ResNet(或是說DenseNet)。我不清楚這個加分題的模型是否要達到準確率的要求才能拿到分，我自己還是有讓它上0.87。

## Demo
- 1.跑一次看accuracy，解釋模型
- 2.depthwise convulotion在幹嘛
    - 主要是透過對捲基層分組(group)，input channel多少group就會固定是多少。讓模型在這一層的CNN不會有channel間的交互運算，達到大幅減少參數量和運算量的效果。
- 3.Batch Normalization如何解決梯度消失或爆炸
    - 透過對梯度做Normalization，避免梯度出現過大或過小的值來解決梯度消失或爆炸
- 4.Leaky ReLU和ELU可以解決ReLU什麼問題
    - Dying ReLU的問題，避免因為負半區的梯度為零不會更新。
- 5.為什麼input feature是736
    - 因為前面的捲基層捲出來的矩陣大小攤平乘起來是這個數字。