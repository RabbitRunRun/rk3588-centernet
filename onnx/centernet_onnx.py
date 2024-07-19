# -*- coding:utf-8 -*
import os.path
import time

import cv2
import numpy as np


class DetModel(object):
    def __init__(self, model_path, input_size=512):
        assert os.path.exists(model_path)
        self.net =  cv2.dnn.readNet(model_path)
        self.input_size = input_size
        self.pre_image = None

    def pad_resize(self, image, input_size):
        h, w, _ = image.shape
        scale =  input_size / max(h, w)
        new_image =  cv2.resize(image, (int(w * scale), int(h * scale)))
        h_, w_, _ = new_image.shape
        pad_x0, pad_y0 = (input_size - w_)//2,  (input_size - h_)//2
        new_image = np.pad(new_image, ((pad_y0, input_size - h_ - pad_y0), (pad_x0, input_size - w_ - pad_x0), (0, 0)), 'constant')
        return new_image, [pad_x0, pad_y0, scale]

    def process(self, image):
        start_time = time.time()
        scale_image, pad_info = self.pad_resize(image, self.input_size)
        scale_image = cv2.cvtColor(scale_image, cv2.COLOR_BGR2GRAY)
        print("scale_image shape:", scale_image.shape)
        if self.pre_image is None:
            self.pre_image = np.stack([scale_image for _ in range(3)], axis=-1)
        else:
            self.pre_image = np.concatenate([self.pre_image[:, :, 1:], scale_image[..., np.newaxis]], axis=-1)
        scale_image = self.pre_image.copy()
        self.net.setInput(cv2.dnn.blobFromImage(scale_image, 1.0, (self.input_size, self.input_size), (0, 0, 0)))
        output = self.net.forward()[0]
        scale = 7.641791044776119
        output[..., :4] *= scale
        pad_x0, pad_y0, scale = pad_info
        output[..., :4] -= [pad_x0, pad_y0, pad_x0, pad_y0]
        output[..., :4] /= scale
        # print(output.shape)
        idx = 0
        for bnd in output:
            x0, y0, x1, y1,  score = bnd
            # crop area
            if score > 0.5:
                crop_image = image[int(max(y0-5, 0)):int(y1+5), int(max(x0-5, 0)):int(x1+5)]
                crop_image = cv2.resize(crop_image, (100, 100))
                image[idx*100:idx*100+100, -100:] = crop_image
                idx += 1
                cv2.putText(image, f"{score:.2f}", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        # time.sleep(0.3)
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)

        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("demo", image)
        cv2.waitKey(0)
        return image

if __name__ == '__main__':
    model = DetModel("center_net_opset12.onnx")
    cap = cv2.VideoCapture("demo.mp4")
    # writer = None
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = model.process(image)








