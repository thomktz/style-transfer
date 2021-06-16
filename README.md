# A Neural Style Transfer implementation

Python, torch

### NST principle

Inputs : *content* image `C`, *style* image `S`

A *generated* image (tensor `G`) is created from either a copy of *content* or from random noise. This tensor is the only trained parameter.  
A pretrained VGG model, or a pretrained Resnet151 model is used to extract features from all three tensors. The features are stored at multiple steps during any forward pass in the model, *L* times

The content and style losses are defined as such :  

[](https://user-images.githubusercontent.com/60552083/122243699-9e51b980-cec4-11eb-8cd3-ca4224b2b8d1.png)


