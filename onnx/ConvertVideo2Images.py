# -*- coding:utf-8 -*
import os.path
import time

import cv2
import numpy as np
import glob

class DetModel(object):
    def __init__(self, model_path, input_size=512):
        assert os.path.exists(model_path)
        self.net =  cv2.dnn.readNet(model_path)
        self.input_size = input_size
        self.pre_image = None
        self.id = 0

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
        # scale_image = cv2.cvtColor(scale_image, cv2.COLOR_BGR2GRAY)
        print("scale_image shape:", scale_image.shape)
        self.pre_image = scale_image
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
                cv2.putText(image, f"{score:.5f}", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        # time.sleep(0.3)
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)

        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("demo", image)
        cv2.waitKey(10)
        # cv2.imwrite("results/" + str(self.id) + ".png", image)
        # self.id = self.id + 1
        return image

def video_to_images(video_file, saved_concated_images_path='images_concate3gray', saved_origin_images_path='images_origin'):
    cap = cv2.VideoCapture(video_file)
    # 连续三帧数据的拼接
    concated_image = None
    id = 1000
    begin_id = id
    saving_nunber = 150
    while True:
        ret, image = cap.read()
        if not ret:
            break
        # bgr2gray
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('gray_image shape:', gray_image.shape)
        if concated_image is None:
            concated_image = np.stack([gray_image for _ in range(3)], axis=-1)
        else:
            concated_image = np.concatenate([concated_image[..., 1:], gray_image[..., np.newaxis]], axis=-1)
        id = id + 1
        if id < begin_id + 2:
            id = id + 1
            continue 
        cv2.imwrite(saved_concated_images_path + "/" + str(id) + ".png", concated_image)
        print("origin image shape:", image.shape)
        cv2.imwrite(saved_origin_images_path + "/" + str(id) + ".png", image)
        saving_nunber = saving_nunber - 1
        if saving_nunber < 0:
            return 


def pad_resize(image, input_size=512):
    h, w, _ = image.shape
    scale =  input_size / max(h, w)
    new_image =  cv2.resize(image, (int(w * scale), int(h * scale)))
    h_, w_, _ = new_image.shape
    pad_x0, pad_y0 = (input_size - w_)//2,  (input_size - h_)//2
    new_image = np.pad(new_image, ((pad_y0, input_size - h_ - pad_y0), (pad_x0, input_size - w_ - pad_x0), (0, 0)), 'constant')
    return new_image, [pad_x0, pad_y0, scale]

def get_quantization_images_from_video(video_file, saved_begin_id = 0, saved_concated_images_path='to_quantization_images'):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频的总帧数: {total_frames}")
    # 连续三帧数据的拼接
    concated_image = None

    saving_nunber = 30
    step = int(total_frames / saving_nunber)
    print('saving step:', step)

    id = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        # bgr2gray
        scale_image, pad_info = pad_resize(image, 512)
        gray_image = cv2.cvtColor(scale_image, cv2.COLOR_BGR2GRAY)
        # print('gray_image shape:', gray_image.shape)
        if concated_image is None:
            concated_image = np.stack([gray_image for _ in range(3)], axis=-1)
        else:
            concated_image = np.concatenate([concated_image[..., 1:], gray_image[..., np.newaxis]], axis=-1)
        if id % step == 0:
            cv2.imwrite(saved_concated_images_path + "/" + str(saved_begin_id) + ".png", concated_image)
            saved_begin_id = saved_begin_id + 1
        id = id + 1

if __name__ == '__main__':
    # model = DetModel("center_net_opset12.onnx")
    images_path = "to_quantization_images/*.png"
    image_files = glob.glob(images_path)
    with open("dataset.txt", "w", encoding='utf-8') as writer:
        for file in image_files:
            file = file.replace('\\', '/')
            writer.write(file + '\n')
    print("image_files:", image_files)
    exit(0)
    # for image_file in image_files:
    #     print(image_file)
    #     image = cv2.imread(image_file)
    #     image = model.process(image)
    #     image_name = image_file[image_file.find('\\')+1:]
    #     cv2.imwrite(os.path.join("results", "result_" + image_name), image)

    # get_quantization_images_from_video(video_file='demo_big.mp4', saved_begin_id=1000000)

    







