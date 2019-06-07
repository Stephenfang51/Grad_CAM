## Grad_CAM （CNN_visualization)
複雜的CNN中使用Grad_CAM技術將CNN最後一層output對圖像提取的特征進行可視化


### Original input

![image](gorilla.jpg)

### 將heating map疊加原圖效果
可以清楚看見網絡是經由黑猩猩的面部特征判別目標物
![image](gorilla_heatmap.jpg)





Enviroment

Python 3.7

### module you need

* Keras with Pretrained_VGG16
* matplotlib
* numpy
* openCV











參考論文：
https://arxiv.org/pdf/1610.02391v1.pdf
