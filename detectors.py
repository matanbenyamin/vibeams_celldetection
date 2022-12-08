import cv2
import numpy as np
import plotly.express as px
from utils import get_frame, plot
import pickle



class TemplateBasedDetector():
    def __init__(self, match_threshold = 0.6, iou_threshold = 0.5):
        self.templates = []
        self.match_threshold = match_threshold
        self.iou_threshold = iou_threshold

    def cell_preprocess(self, crop):
        crop = cv2.equalizeHist(crop)
        return crop

    def get_templates(self, N=10, vid_path=r'C:\Users\lab7\Downloads\Jurkat_3_75M_x10_starScan.mp4'):
        N = 10
        templates = []
        cap = cv2.VideoCapture(vid_path)
        for i in range(N):
            img = get_frame(vid_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r = cv2.selectROI(img, False)
            templates.append(img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
        cv2.destroyWindow('ROI selector')
        # delete empty templates
        self.templates = [t for t in templates if t.size != 0]
        return templates

    def get_cell_size(self, img, bboxes):

        wdiff = 1
        wcorr = 0
        w = 15

        crop = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        crop = crop.copy()
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
        crop = cv2.Canny(crop, 10, 200)
        crop = cv2.GaussianBlur(crop, (5, 5), 0)

        radii = []
        centers = []

        for b in bboxes:
            print('crop #', b)
            crop3 = crop[b[1]-w:b[3]+w, b[0]-w:b[2]+w].copy()
            mask = np.zeros((int(6 + crop3.shape[0]), int(6 + crop3.shape[1])), dtype=np.float32)
            # add zeros around crop3 so it will be the same size as mask
            crop3 = np.pad(crop3, ((np.floor(mask.shape[0] - crop3.shape[0]).astype(int) // 2,
                                    np.ceil(mask.shape[0] - crop3.shape[0]).astype(int) // 2),
                                   (np.floor(mask.shape[1] - crop3.shape[1]).astype(int) // 2,
                                    np.ceil(mask.shape[1] - crop3.shape[1]).astype(int) // 2)),
                           'constant', constant_values=0)
            # if the sizes are not the same, add zeros to the right and bottom
            if crop3.shape[0] != mask.shape[0]:
                crop3 = np.pad(crop3, ((0, mask.shape[0] - crop3.shape[0]), (0, 0)), 'constant', constant_values=0)
            if crop3.shape[1] != mask.shape[1]:
                crop3 = np.pad(crop3, ((0, 0), (0, mask.shape[1] - crop3.shape[1])), 'constant', constant_values=0)
            crop3 = crop3 / np.sum(crop3)
            best = -np.inf
            for r in range(7, 60, 5):
                for x in range(0, crop3.shape[0], 5):
                    for y in range(0, crop3.shape[1], 5):
                        for invert in [False, True]:

                            # get the circle
                            mask[:] = 0
                            cv2.circle(mask, (y, x), r, (255, 255, 255), 1)
                            mask = cv2.GaussianBlur(mask, (23, 23), 0)
                            mask = mask / np.sum(mask)

                            # correlate the crop with the circle mask
                            if invert:
                                crop3 = 255 - crop3
                            # corr = cv2.matchTemplate(crop3.astype(np.float32), mask, cv2.TM_CCOEFF_NORMED)

                            err = -0.5 * wdiff * np.sum(np.abs(crop3 - mask)) + wcorr * cv2.matchTemplate(
                                crop3.astype(np.float32), mask,
                                cv2.TM_CCOEFF_NORMED)

                            # if np.max(abs(corr)) > best:
                            if err > best:
                                # best = np.max(abs(corr))
                                best = err
                                best_mask = mask.copy()
                                best_rxy = [r, x, y]

            radii.append(best_rxy[0])
            centers.append([best_rxy[1] + b[1] - w, best_rxy[2] + b[0] - w])
        return radii, centers


    def detect(self, img):
        img = self.cell_preprocess(img)

        # preprocess template
        templates = [self.cell_preprocess(t) for t in self.templates]

        bboxes = []
        best_template = []
        scales = []
        corrs = []
        for scale in np.linspace(0.65, 1.5, 7):
            for template in templates:
                # resacale the template
                template = cv2.resize(template, (0, 0), fx=scale, fy=scale)

                # apply template matching to find the template in the image
                match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                # threshold the match image
                loc = np.where(match >= self.match_threshold)
                # draw a rectangle around the matched region
                for pt in zip(*loc[::-1]):
                    bboxes.append([pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]])
                    best_template.append(template)
                    scales.append(scale)
                    corrs.append(match[pt[1], pt[0]])

        bboxes = np.array(bboxes)
        best_template = np.array(best_template)
        scales = np.array(scales)
        corrs = np.array(corrs)


        pred = dict()
        pred['bboxes'] = bboxes
        pred['best_template'] = best_template
        pred['scales'] = scales
        pred['corrs'] = corrs
        self.pred = pred
        return pred

    def intersection(self, bbox1, bbox2):
        # loope over all bboxes and calculate the iou
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        x_overlap = max(0, min(x2, x4) - max(x1, x3))
        y_overlap = max(0, min(y2, y4) - max(y1, y3))
        intersection = x_overlap * y_overlap
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        return intersection / np.min([area1, area2])

    def nms(self, pred, overlapThresh=0.05):

        bboxes = pred['bboxes']

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = y2.argsort()[::-1]
        keep = []
        large_boxes = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= overlapThresh)[0]
            order = order[inds + 1]

        bboxes = bboxes[keep]
        pred['bboxes'] = bboxes
        pred['best_template'] = pred['best_template'][keep]
        pred['scales'] = pred['scales'][keep]
        pred['corrs'] = pred['corrs'][keep]

        return pred

    def cell_color_filter(self, img, pred):
        # this filter is used to remove false positives that are not cells
        # it does this by checking the color of the cell and removing cells that are not of this color
        # the color is determined by the color of the cell in the template image
        #the backgrounf color is determined by the average color outside all bbbxes


        # calculate cell color histogram
        cell_color_hist = []
        for bbox in bboxes:
            hist = np.histogram(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], bins=5, range=(0, 256))[0]
            # normalize the histogram
            hist = hist / np.sum(hist)
            cell_color_hist.append(hist)
        cell_color_hist = np.array(cell_color_hist)
        average_cell_color = np.mean(cell_color_hist, axis=0)

        keep_inds = []
        for i, cell_hist in enumerate(cell_color_hist):
            # calculate the distance between the cell color and the background color
            dist = np.linalg.norm(cell_hist - average_cell_color)
            print(dist)
            # if the distance is small then the cell is probably not a cell
            if dist < 0.5:
                keep_inds.append(i)

        bboxes = bboxes[keep_inds]
        pred['bboxes'] = bboxes
        pred['best_template'] = pred['best_template'][keep_inds]
        pred['scales'] = pred['scales'][keep_inds]
        pred['corrs'] = pred['corrs'][keep_inds]



        # calculate the distance between the cell color and the background color


        # remove cells that are not of the same color as the template
        # this is done by calculating the euclidean distance between the cell color and the template color
        # and removing cells that are further away than a threshold




    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    def load(self, path):
        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self




