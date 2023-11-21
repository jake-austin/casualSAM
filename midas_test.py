import torch
import numpy as np
import cv2
import urllib.request

import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

midas_v3 = torch.hub.load("intel-isl/MiDaS", 'DPT_Large')
# midas_v3.eval()
midas_v3_1 = torch.hub.load("intel-isl/MiDaS", 'DPT_BEiT_L_512')
# midas_v3_1.eval()


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

img1 = cv2.imread(filename)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img1_downsampled = cv2.resize(img1, (384, 192))

_img2 = cv2.imread('/home/jake-austin/DAVIS/JPEGImages/480p/lady-running/00000.jpg')
_img2 = cv2.cvtColor(_img2, cv2.COLOR_BGR2RGB)

img2 = np.load('/home/jake-austin/casualSAM/experiment_logs_midasv3/11-02/davis_dev/lady-running/init_window_BA/0000.npz')['img']
img2 = (img2 * 255).astype(np.uint8)

input_batch1 = transform(img1)
input_batch1_resized = transform(img1_downsampled)
input_batch2 = transform(img2)

with torch.no_grad():
    prediction1_resized_v3 = midas_v3(input_batch1_resized)
    prediction1_v3 = midas_v3(input_batch1)
    prediction2_v3 = midas_v3(input_batch2)
    prediction1_resized_v3_1 = midas_v3_1(input_batch1_resized)
    prediction1_v3_1 = midas_v3_1(input_batch1)
    prediction2_v3_1 = midas_v3_1(input_batch2)

    prediction1_resized_v3 = torch.nn.functional.interpolate(
        prediction1_resized_v3.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    prediction1_v3 = torch.nn.functional.interpolate(
        prediction1_v3.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    prediction2_v3 = torch.nn.functional.interpolate(
        prediction2_v3.unsqueeze(1),
        size=img2.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    prediction1_resized_v3_1 = torch.nn.functional.interpolate(
        prediction1_resized_v3_1.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    prediction1_v3_1 = torch.nn.functional.interpolate(
        prediction1_v3_1.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    prediction2_v3_1 = torch.nn.functional.interpolate(
        prediction2_v3_1.unsqueeze(1),
        size=img2.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()


plt.imshow(prediction1_resized_v3)
plt.savefig('prediction1_resized_v3.png')
plt.imshow(prediction1_v3)
plt.savefig('prediction1_v3.png')
plt.imshow(prediction2_v3)
plt.savefig('prediction2_v3.png')
plt.imshow(prediction1_resized_v3_1)
plt.savefig('prediction1_resized_v3_1.png')
plt.imshow(prediction1_v3_1)
plt.savefig('prediction1_v3_1.png')
plt.imshow(prediction2_v3_1)
plt.savefig('prediction2_v3_1.png')
