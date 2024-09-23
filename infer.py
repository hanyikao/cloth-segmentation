import time
start_time = time.time()
import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_seg_model

device = "cuda"

image_dir = "../outerwear" #"../imaterialist/test"
result_dir = "results/training_cloth_segm_u2net_exp1/images"
# segment_dir = "output_masks"
# checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
checkpoint_path = "results/training_cloth_segm_u2net_exp1/checkpoints/itr_00097000_u2net.pth"
do_palette = True


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

def get_legend():
    class_to_color = dict()
    class_name = ['background', 'tops', 'sweater', 'outerwear', 'bottoms', 'wholebody']
    for idx, i in enumerate(range(len(palette) // 3)):
        class_to_color[idx] = (palette[i*3] / 255., palette[i*3+1] / 255., palette[i*3+2] / 255.)
    # Create a legend
    legend_elements = [plt.Line2D([], [], marker='s', color='w', markerfacecolor=color, markersize=10, linewidth=0, label=f"{class_name[class_label]}")
                    for class_label, color in class_to_color.items()]
    return legend_elements


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = load_seg_model(checkpoint_path, device)
palette = get_palette(6)



if __name__ == "__main__":
    images_list = sorted(os.listdir(image_dir))
    pbar = tqdm(total=len(images_list))
    for idx, image_name in enumerate(images_list):
        if not image_name.endswith('.jpg'): continue
        img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        w, h = img.size
        img_resize = img.resize((768, 768), Image.BICUBIC)
        image_tensor = transform_rgb(img_resize)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor_list = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor_list[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        output = Image.fromarray(output_arr.astype("uint8"), mode="L")
        output = output.resize((w, h), Image.BILINEAR)
        if do_palette:
            output.putpalette(palette)
        output = output.convert("RGB")

        # output.save(os.path.join(segment_dir, image_name[:-4] + ".png"))
        overlayed = Image.blend(img, output, 0.3)
        # overlayed.save(os.path.join(result_dir, image_name[:-4] + ".png"))
        combined = Image.new('RGB', (img.width + overlayed.width + output.width, max(img.height, overlayed.height, output.height)))
        combined.paste(img, (0, 0))
        combined.paste(overlayed, (img.width, 0))
        combined.paste(output, (img.width + overlayed.width, 0))
        combined.save(os.path.join(result_dir, image_name[:-4] + ".png"))

        # fig, ax = plt.subplots(figsize=(w, h))
        # ax.imshow(np.array(combined))
        # ax.axis('off')  # Hide axes
        # # Add a legend
        # plt.legend(handles=get_legend(), loc='upper right', handlelength=0.7, handleheight=1)
        # plt.imsave(os.path.join(result_dir, image_name[:-4] + ".png"))

        pbar.update(1)
        # if idx == 0: break

    pbar.close()
    print(time.time() - start_time)