---
share: true
---
This summary is based off this [blog](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607). 

We basically augment the data (image for example) by cropping, resizing, recoloring, etc. Then we ask the model to give a similar output for the ones that comes out of the same original image, and discriminate ones that does not come out of the same image. As augmentation does not change the subject of interest, the hope is that the model learn some structure information from it. 

After that, the trained model can be used for [[semi-supervised learning|semi-supervised learning]] or other stuff.