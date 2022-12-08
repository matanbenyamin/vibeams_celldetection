#import necessary library
import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread(r"/home/vk/Desktop/EdgeDetection-CV-main/3.png")
(H, W) = img.shape[:2]
#Download prototxt and pretrained caffe model - use my github
protoPath = r"/home/vk/Desktop/EdgeDetection-CV-main/hed-edge-detector-master/deploy.prototxt"
modelPath = r"/home/vk/Desktop/EdgeDetection-CV-main/hed-edge-detector-master/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

cd C:\Users\lab7\PycharmProjects\celldetection\Traditional-and-Deeplearning-based-EdgeDetection\hed-edge-detector-master

python edge.py --input C:\Users\lab7\Downloads\630_Moment2.jpg --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel


import torch
import kornia
import cv2
import numpy as np
from detectors import TemplateBasedDetector
from utils import get_frame, plot


import matplotlib.pyplot as plt

# read the image with OpenCV
img = get_frame( r'C:\Users\lab7\Downloads\Jurkat_3_75M_x10_starScan.mp4')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7, 7), 0)

plot(img)
# convert to torch tensor
data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)/255.  # BxCxHxW
# create the operator
canny = kornia.filters.Canny()

# blur the image
x_magnitude, x_canny = canny(data.float())

cann = x_canny.detach().numpy()
cann = np.squeeze(cann)
plot(cann)

crop = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
crop3 = cv2.Canny(crop, 10, 200)
# crop3 = cv2.GaussianBlur(crop3, (5, 5), 0)
plot(crop3)


# convert back to numpy
img_magnitude: np.ndarray = kornia.tensor_to_image(x_magnitude.byte())
img_canny: np.ndarray = kornia.tensor_to_image(x_canny.byte())

# Create the plot
fig, axs = plt.subplots(1, 3, figsize=(16,16))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('image source')
axs[0].imshow(img)

axs[1].axis('off')
axs[1].set_title('canny magnitude')
axs[1].imshow(img_magnitude, cmap='Greys')

axs[2].axis('off')
axs[2].set_title('canny edges')
axs[2].imshow(img_canny, cmap='Greys')

plt.show()