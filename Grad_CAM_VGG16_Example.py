#導入需要用到的庫
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

 
model = VGG16(weights='imagenet')
model.summary()


img_path = "/Users/stephenfang/PycharmProjects/keras/image/gorilla.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img) #將圖像轉換成數組
x = np.expand_dims(x, axis=0)
'''
Keras一般運行批次圖像,  所有np.expand為X增加first dimension, axis為0表示第一維
(samples, height, width, channels)
'''

x = preprocess_input(x)
'''
preprocess的過程是必須的， 將轉換為model所需要的格式
'''

#讀取圖像x 進行目標預測
preds = model.predict(x)
print("predicted:", decode_predictions(preds, top=3)[0]) #返回top3概率較高的預測目標

#argmax找出preds向量中概率最大的值的索引 Let’s find out the index of the gorilla in the prediction vector
np.argmax(preds[0])

goriila_output = model.output[:, 366] #取出
#返回列表中的第[:, 386]的张量


### 以下进行Grad_CAM操作
last_conv_layer = model.get_layer('block5_conv3')  #取出最後一層feature map

grads = K.gradients(gorilla_output, last_conv_layer.output)[0]  # 計算出gorilla_ouput 基於 最後一層的 梯度
 
pooled_grads = K.mean(grads, axis=(0, 1, 2))  #计算该梯度对每个维度的均值
 
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) 
#第一个参数放入 model的input
#第二个参数ouput输出有两個，one for final layer.output,  second for softmax output后的last layer.output

 
pooled_grads_value, conv_layer_output_value = iterate([x]) 


for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i] 

#建構heat map, 為了便利可視化， 將圖片normalize to 0 - 1
heatmap = np.mean(conv_layer_output_value, axis=-1) #np.mean將array每個元素相加除以個數
heatmap = np.maximum(heatmap, 0)   #np.maximum(x, y) x與y比較取最大， 也就是說heatmap>1的被留下
heatmap /= np.max(heatmap)#np.max 從list找最大值
plt.matshow(heatmap)
plt.show

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  #resize跟原圖大小一樣 640*545
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
final_img = heatmap * 0.4 + img #將heatmap重疊原圖
cv2.imwrite('/Users/stephenfang/PycharmProjects/keras/image/gorilla_heatmap.jpg', final_img)

