import os
import copy
import cv2
import numpy as np
from collections import OrderedDict

import torch


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def save_checkpoint(model, save_path):
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def save_checkpoints(opt, itr, net):
    save_checkpoint(
        net,
        os.path.join(opt.save_dir, "checkpoints", "itr_{:08d}_u2net.pth".format(itr)),
    )


def check_or_download_model(file_path):
    import gdown
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    from networks import U2NET
    net = U2NET(in_ch=3, out_ch=6)
    check_or_download_model(checkpoint_path)
    # net = load_checkpoint_mgpu(net, checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    return net