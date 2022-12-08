import numpy as np
import cv2 as cv
import plotly.express as px


def get_frame(vid_path, frame_idx = None):
    cap = cv.VideoCapture(vid_path)
    if frame_idx is None:
        frame_idx = np.random.randint(0,  cap.get(cv.CAP_PROP_FRAME_COUNT))
    vid = cv.VideoCapture(vid_path)
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = vid.read()
    return frame


def plot(frame, bboxes = None, txt = None):
    frame = frame.copy()
    if bboxes is not None:
        for inds, bbox in enumerate(bboxes):
            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if txt is not None:
                cv.putText(frame, txt[inds], (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    fig = px.imshow(frame)
    fig.show(renderer="browser")
    return fig


