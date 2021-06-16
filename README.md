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

