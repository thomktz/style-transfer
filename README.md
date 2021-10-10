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

### *Starry Night* by Van Gogh, and a town
![starrytown](https://user-images.githubusercontent.com/60552083/122601543-bc105180-d071-11eb-9824-6e2f751ec5b9.png)
![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/60552083/122601550-bdda1500-d071-11eb-9c82-088890d407c3.gif)
![0059](https://user-images.githubusercontent.com/60552083/122601560-c4688c80-d071-11eb-9c16-cd3077381323.png)


### *The Scream*, by Edvard Munch and the Windows background image
![images2](https://user-images.githubusercontent.com/60552083/122255168-27b9b980-cece-11eb-9342-61fe0592ba52.png)
![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/60552083/122588107-f9b7af00-d05e-11eb-8ee7-2c0f531a1440.gif)
![0030](https://user-images.githubusercontent.com/60552083/122588170-0fc56f80-d05f-11eb-9f77-03fd795d68a4.png)


### *Tableau I*, 1921, by Piet Mondrian and a picture of the London skyline
![skylinemandrian](https://user-images.githubusercontent.com/60552083/122271451-b551d500-cedf-11eb-90e6-3ad75282bada.png)
![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/60552083/122271610-e29e8300-cedf-11eb-9c8f-a144a0fa6e89.gif)
![01980 (1)](https://user-images.githubusercontent.com/60552083/122271657-f0ec9f00-cedf-11eb-88b4-1359c1960f02.png)

### *Emergence of Orange*, by Koola Adams and a Tokyo street
![koolatokyo](https://user-images.githubusercontent.com/60552083/122280666-b982f000-cee9-11eb-85e4-01eef9a5e13e.png)
![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/60552083/122280722-c7d10c00-cee9-11eb-8c5c-2d630aa7f5df.gif)
![02100](https://user-images.githubusercontent.com/60552083/122280772-d4556480-cee9-11eb-8ce8-3eea6e652f04.png)

### *Emergence of Orange*, by Koola Adams and a city skyline
![koolaskyline](https://user-images.githubusercontent.com/60552083/122293483-5730ec00-cef7-11eb-9532-d0ab33ac548e.png)
![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/60552083/122293499-5d26cd00-cef7-11eb-9ceb-fedb6cade3dd.gif)
![02015](https://user-images.githubusercontent.com/60552083/122293581-72036080-cef7-11eb-94b1-99ac2289e21d.png)
