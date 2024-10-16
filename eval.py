import numpy as np
import torch
import time
import os
from data.base_dataset import Denormalize_image

def evaluation(model, dataloader, loss_func, opt, itr=0, start_time=0, device="cuda:0", scaler_list=[]):
    import pprint
    from tqdm import tqdm
    from PIL import Image
    from infer import get_palette
    from utils.metric import SegmentationMetrics
    import matplotlib.pyplot as plt

    palette = get_palette(6)
    model.eval()
    loss_list, metric_list = [], []
    with torch.no_grad():
        idx = 0
        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.type(torch.long)
            label = label.to(device)

            d0, d1, d2, d3, d4, d5, d6 = model(image)
            
            loss0 = loss_func(d0, label)
            loss1 = loss_func(d1, label)
            loss2 = loss_func(d2, label)
            loss3 = loss_func(d3, label)
            loss4 = loss_func(d4, label)
            loss5 = loss_func(d5, label)
            loss6 = loss_func(d6, label)
            total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            loss_list.append([total_loss.item(), loss0.item()])
            
            metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
            _, dice, precision, recall = metric_calculator(label, d0)
            metric_list.append([dice, precision, recall])


            output_tensor = torch.nn.functional.log_softmax(d0, dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=False)[1]
            output_tensor = output_tensor[-1]
            output_arr = output_tensor.cpu().numpy()

            output = Image.fromarray(output_arr.astype("uint8"), mode="L")
            output.putpalette(palette)
            output = output.convert("RGB")

            img = Image.fromarray(Denormalize_image(opt.mean, opt.std)(image.cpu())[-1].numpy().transpose(1,2,0)).convert('RGB')
            overlayed = Image.fromarray(label[-1].cpu().numpy().astype("uint8"), mode="L")
            overlayed.putpalette(palette)
            overlayed = overlayed.convert("RGB")
            combined = Image.new('RGB', (img.width + overlayed.width + output.width, max(img.height, overlayed.height, output.height)))
            combined.paste(img, (0, 0))
            combined.paste(overlayed, (img.width, 0))
            combined.paste(output, (img.width + overlayed.width, 0))
            combined.save(os.path.join(opt.save_dir, "images", str(idx) + ".png"))
            idx += 1



        total_loss, loss0 = np.mean(loss_list, axis=0)
        dice, precision, recall = np.mean(metric_list, axis=0)
        pprint.pprint(
            "[step-{:08d}] [time-{:.3f}] [val_total_loss-{:.6f}] [val_loss0-{:.6f}] [val_dice-{:.4f}] [val_precision-{:.4f}] [val_recall-{:.4f}]".format(
                itr, time.time() - start_time, total_loss, loss0, dice, precision, recall
            )
        )
        scaler_list.extend([total_loss, loss0, dice, precision, recall])


if __name__ == '__main__':
    from data.custom_dataset_data_loader import CustomTestDataLoader
    from utils.saving_utils import load_seg_model
    # from options.base_options import parser
    from options.argparse_opt import parser
    from utils.distributed import set_seed
    
    start_time = time.time()
    opt = parser().get_options()
    device = torch.device("cuda:0")
    checkpoint_path = "results/training_cloth_segm_u2net_exp5/checkpoints/itr_00021000_u2net.pth"
    u_net = load_seg_model(checkpoint_path, device)
    u_net = u_net.to(device)
    
    set_seed(1000)

    custom_dataloader = CustomTestDataLoader()
    custom_dataloader.initialize(opt)
    validloader = custom_dataloader.get_loader()

    # loss function
    weights = np.array([1, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = torch.nn.CrossEntropyLoss(weight=weights).to(device)

    evaluation(u_net, validloader, loss_CE, opt, start_time=start_time, device=device)
    print("Exiting..............")
