import os
import numpy as np
import cv2
from rknn.api import RKNN
import time
import glob
from logdown import logdown

ONNX_MODEL = 'center_net_opset19.onnx'
RKNN_MODEL = './rknn_models/center_net_mmse_u8.rknn'
QUANT_IMG_RGB2BGR = True
DATASET = './dataset-mmse.txt'

QUANTIZE_ON = True
BOX_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE = (512, 512) 

_force_builtin_perm = False
rknn = None
def convert_model():

    # Create RKNN object
    global rknn
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)


    # pre-process config
    print('--> Config model')
    # rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR, 
    #             quantized_algorithm='kl_divergence')
    # rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR, 
    #         quantized_algorithm='normal')
    rknn.config(target_platform = 'rk3588', quant_img_RGB2BGR=QUANT_IMG_RGB2BGR, 
            quantized_algorithm='mmse')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

def pad_resize(image, input_size=512):
    h, w, _ = image.shape
    scale =  input_size / max(h, w)
    new_image =  cv2.resize(image, (int(w * scale), int(h * scale)))
    h_, w_, _ = new_image.shape
    pad_x0, pad_y0 = (input_size - w_)//2,  (input_size - h_)//2
    new_image = np.pad(new_image, ((pad_y0, input_size - h_ - pad_y0), (pad_x0, input_size - w_ - pad_x0), (0, 0)), 'constant')
    return new_image, [pad_x0, pad_y0, scale]

def test_model(image_path, image_name):
    # Set inputs
    image = cv2.imread(os.path.join(image_path, image_name))

    # image preprocessing

    start_time = time.time()
    scale_image, pad_info = pad_resize(image, input_size=512)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[scale_image], inputs_pass_through=[0 if not _force_builtin_perm else 1])
    print("outputs len:", len(outputs))
    output = outputs[0]
    print('output shape:', output.shape)
    output = output.squeeze()
    print('after squeeze output shape:', output.shape)

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
            print(f'location:[x0,y0,x1,y1]=[{int(max(x0-5, 0))},{int(max(y0-5, 0))},{int(x1+5)},{int(y1+5)}]')
            crop_image = image[int(max(y0-5, 0)):int(y1+5), int(max(x0-5, 0)):int(x1+5)]
            crop_image = cv2.resize(crop_image, (100, 100))
            image[idx*100:idx*100+100, -100:] = crop_image
            idx += 1
            cv2.putText(image, f"{score:.5f}", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    # time.sleep(0.3)
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)

    saved_path = RKNN_MODEL[RKNN_MODEL.rfind('/') + 1:] + '_' + 'simu_results'
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
    cv2.imwrite(os.path.join(saved_path, "result_" + image_name), image)


def rknn_debug_with_simulator_debug_analysis():
    rknn_debug_txts = glob.glob('rknn_debug/*.txt')
    debug_txts = glob.glob('debug/*.txt')
    # print(rknn_debug_txts)
    # print(debug_txts)
    # print(len(rknn_debug_txts))
    # print(len(debug_txts))
    assert len(rknn_debug_txts) == len(debug_txts)
    for idx in range(len(rknn_debug_txts)):
        debug_data = np.loadtxt(f'debug/output{idx}.txt').flatten()
        # print('debug_data shape:', debug_data.shape)
        rknn_debug_data = np.loadtxt(f'rknn_debug/output{idx}.txt').flatten()
        # print('rknn_debug_dat shape:', rknn_debug_data.shape)
        assert debug_data.shape == rknn_debug_data.shape
        difference = debug_data - rknn_debug_data
        print(f'difference{idx} max:', np.max(difference))
        print(f'difference{idx} min:', np.min(difference))
        print(f'difference{idx} mean:', np.mean(difference))
        print(f'difference{idx} var:', np.var(difference))

def test_model1(image_path, image_name):
    # Set inputs
    image = cv2.imread(os.path.join(image_path, image_name))

    # image preprocessing

    start_time = time.time()
    scale_image, pad_info = pad_resize(image, input_size=512)
    cv2.imwrite("input_to_center_net.png", scale_image)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[scale_image], inputs_pass_through=[0 if not _force_builtin_perm else 1])
    print("outputs len:", len(outputs))
    for idx in range(len(outputs)):
        np.savetxt(f"debug/output{idx}.txt", outputs[idx].flatten())
    
    exit(0)
    output = outputs[0]
    print('output shape:', output.shape)
    output = output.squeeze()
    print('after squeeze output shape:', output.shape)
    np.savetxt("center_net_result.txt", output)
    
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

    saved_path = RKNN_MODEL[:RKNN_MODEL.find('.')] + '_' + 'simu_results'
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
    cv2.imwrite("result_" + image_name, image)
    exit(0)

if __name__ == '__main__':
    with logdown():
        # load and build model
        convert_model() 

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # test images using simulator
        image_files = glob.glob('concated3gray_images/*.png')
        id = 0
        for image_file in image_files:
            # print('Processing ', image_file, '...')
            path = image_file[:image_file.find('/')]
            # print('image path: ', path)
            image_name = image_file[len(path)+1:]
            # print('image name: ', image_name)
            # test model
            if id % 10 == 0:
                test_model(path, image_name=image_name)
    
    # # release rknn
    # rknn.release()
    # rknn_debug_with_simulator_debug_analysis()

