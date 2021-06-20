# %%
from model import RESNET, VGG
from image_treatment import prepare_image, out_size, undo_transform
import torch
import tqdm
from torchvision.utils import save_image

device = torch.device('cuda')
torch.cuda.empty_cache()
source_path = "source_images/"

content_path = source_path + "town.jpg"
style_path = source_path + "klimt.jpg"

content = prepare_image(content_path).to(device)
style = prepare_image(style_path).to(device)

generated = content.clone().requires_grad_(True)
#generated = torch.randn(content.shape, device = device, requires_grad=True)

#model = RESNET().to(device).eval()
model = VGG().to(device).eval()

steps = 45
lr = 0.02
alpha = 100.0
beta = 0.1
delta = 0
optimizer = torch.optim.LBFGS([generated], lr=0.2)

for step in tqdm.notebook.tqdm(range(steps)):
    if step % 1 == 0:
        #print(f"Step {step}, total loss = {total_loss.item()}")
        save_image(undo_transform(generated), f"out15/{str(step).zfill(4)}.png")
    def closure():
        generated_features = model(generated)
        style_features = model(style)
        content_features = model(content)
        style_loss, content_loss, reg_loss = 0, 0, 0
        for gen_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
            batch_size, channel, h, w = gen_feature.shape
            content_loss += torch.mean((content_feature - gen_feature)**2)
            
            G = gen_feature.view(channel, w*h).mm(gen_feature.view(channel, w*h).t())
            A = style_feature.view(channel, w*h).mm(style_feature.view(channel, w*h).t())
            
            style_loss += torch.mean((G-A)**2)

            reg_loss += torch.sum(torch.abs(gen_feature[:, :, :, :-1] - gen_feature[:, :, :, 1:])) + torch.sum(torch.abs(gen_feature[:, :, :-1, :] - gen_feature[:, :, 1:, :]))
        total_loss = alpha * content_loss + beta * style_loss + delta * reg_loss
        optimizer.zero_grad()
        total_loss.backward()
        return total_loss
    optimizer.step(closure)

  