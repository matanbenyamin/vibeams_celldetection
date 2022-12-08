from detectors import TemplateBasedDetector
from utils import get_frame
from utils import plot
import cv2
import numpy as np

detector = TemplateBasedDetector()

# ============= initialize the detector
# detector.get_templates(vid_path=r'C:\Users\lab7\Downloads\Jurkat_3_75M_x10_starScan.mp4')
# load
detector = detector.load('detector.pkl')

# ============= detector params


# ============= get a frame
img = get_frame( r'C:\Users\lab7\Downloads\Jurkat_3_75M_x10_starScan.mp4')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ============= detect cells
pred = detector.detect(img)
bboxes = pred['bboxes']



# ============= filter bbox
pred = detector.nms2(pred)

# ============= plot the results
plot(img, pred['bboxes'])

# ============= create mask
centers, radii = detector.get_cell_size(img, pred['bboxes'])
mask = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
for i in range(len(centers)):
    mask = cv2.circle(mask,centers[i], radii[i], (1, 87,155), 2)
    mask = cv2.circle(mask, centers[i], radii[i], (2, 136, 209),-1)
output = np.stack([img, img, img], axis=2)
output = cv2.addWeighted(output, 1, mask, 0.2, 0)
plot(output)
# ============= save
detector.save('detector.pkl')
