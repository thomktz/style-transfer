# %%
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms.transforms import Resize

out_size = 400

def prepare_image(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize((out_size,out_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch

def undo_transform(tensor):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])

    inv_tensor = invTrans(tensor)
    return inv_tensor