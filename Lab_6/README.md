# Lab 6
## Requirements
- Implement a conditional DDPM
    - Choose your conditional DDPM setting.
    - Design your noise schedule and UNet architecture
    - Choose your loss functions.
    - Implement the training function, testing function, and data loader.
-  Generate the synthetic images
    - Evaluate the accuracy of test.json and new test.json.
    - Show the synthetic images in grids for two testing files.
    - Plot the progressive generation process for generating an image.

## Datasets
- 要根據圖片資料集跟condition去輸出指定圖片，並透過預訓練好的模型去對你生成的圖片進行分類，若分類準確率高即可得分。
## Methods
- 這次作業可以使用任何形式的Library。
- 我的model.py和dataloader.py主要是參考網路上github的，詳細在report中有。