# A Neural Style Transfer implementation

Python, torch

### NST principle

Inputs : *content* image `C`, *style* image `S`

A *generated* image (tensor `G`) is created from either a copy of *content* or from random noise. This tensor is the only trained parameter.  
A pretrained VGG model, or a pretrained Resnet151 model is used to extract features from all three tensors. The features are stored at multiple steps during any forward pass in the model, *L* times

The content and style losses are defined as such :  

![](https://user-images.githubusercontent.com/60552083/122243699-9e51b980-cec4-11eb-8cd3-ca4224b2b8d1.png)

And the final loss to backpropagate is  

![](https://user-images.githubusercontent.com/60552083/122244297-10c29980-cec5-11eb-945b-06eb937b3dd5.png)


# Results

### *The Scream*, by Edvard Munch and the Windows background image
6000 steps, ~30 minutes on Google Colab GPU
![images2](https://user-images.githubusercontent.com/60552083/122255168-27b9b980-cece-11eb-9342-61fe0592ba52.png)
![ezgif-3-0dca9815969c](https://user-images.githubusercontent.com/60552083/122255183-2b4d4080-cece-11eb-8b4e-08246a8078d8.gif)
![05980](https://user-images.githubusercontent.com/60552083/122255202-2f795e00-cece-11eb-94bc-f89d2f0f941c.png)

### *Tableau I*, 1921, by Piet Mondrian and a picture of a skyline
2000 steps, 9 minutes on Google Colab GPU
![skylinemandrian](https://user-images.githubusercontent.com/60552083/122271451-b551d500-cedf-11eb-90e6-3ad75282bada.png)
![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/60552083/122271610-e29e8300-cedf-11eb-9c8f-a144a0fa6e89.gif)
![01980 (1)](https://user-images.githubusercontent.com/60552083/122271657-f0ec9f00-cedf-11eb-88b4-1359c1960f02.png)
