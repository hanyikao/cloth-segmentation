import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# Load the SAM model
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda:0")
# predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    min_mask_region_area=1000,  # Requires open-cv to run post-processing
)

# Load and preprocess your image
image = cv2.imread("/home/hykao/Taelor/Archive/outerwear/images/8183515381973/0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

image = cv2.resize(image, (w // 11, h // 11), interpolation=cv2.INTER_AREA)


# predictor.set_image(image)

# # Provide a prompt (e.g., a point or bounding box)
# input_point = np.array([[100, 100]])
# input_label = np.array([1])

# # Get the segmentation mask
# masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())

# # Process the mask as needed
# # ...

# # Save or use the mask for further processing
# cv2.imwrite("path/to/save/mask.png", masks[0])

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
