# %%
import cv2
import glob
from image_treatment import prepare_image, undo_transform
from torchvision.utils import save_image


def make_video():
    img_array = []
    source_path = "generated_images"
    for filename in glob.glob(source_path+'/*.jpeg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)    
    out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def make_images(content_name, style_name):
    source_path = "source_images/"
    content_path = source_path + content_name
    style_path = source_path + style_name

    content = prepare_image(content_path)
    style = prepare_image(style_path)
    
    out_content = undo_transform(content)
    out_style = undo_transform(style)
    
    save_image(out_content, "treated_images/" + content_name)
    save_image(out_style, "treated_images/" + style_name)
# %%
