#include "moving_object_detector.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"
#include "detector_utils.h"

#ifdef SEETA_AUTO_ENCRYPT
#include "SeetaLANLock.h"
#include "hidden/SeetaLockFunction.h"
#include "hidden/SeetaLockVerifyLAN.h"
#endif

namespace seeta {
    // moving object detector class
    class MODetector {
        public:
            explicit MODetector(){
                #ifdef SEETA_AUTO_ENCRYPT
                    SeetaLock_VerifyLAN verify(1001);
                    SeetaLockSafe_call(&verify);
                    
                    SeetaLock_Verify_Check_Instances verify_instances("seeta.detector");
                    SeetaLockSafe_call(&verify_instances);
			    #endif

            };
            int init(const char* model, float shift_scale, int debug = 0);
            ~MODetector();

            void set_score_thresh(float thresh) {
                _score_thresh = thresh;
            }

            float get_score_thresh() {
                return _score_thresh;
            }

            // pass_through=true means image data does not run bgr2gray converting to concate
            //              image_data has 3 channels: resize->copy make border->input to model
            // pass_through=false means image data does run bgr2gray converting and concate
            //              image_data has 3 channels: resize->copy make border->bgr2gray->concate to 3 chanels->input to model
            detect_result_group detect(const char* image_data, 
                                int image_width, int image_height, int pass_through = 0);

            void reset();
        private:
            unsigned char* _model_data;
            int _model_data_size = 0;
            rknn_context _ctx;
            rknn_input_output_num _io_num;
            int _model_input_height = 0;
            int _model_input_width = 0;
            int _model_input_channels = 3;
            float _score_thresh=0.5f;
            // model input data
            cv::Mat _pre_image;
            const float _down_scale = 7.641791044776119f;
            // outputs
            std::vector<detect_result> _inner_results;
            // output size
            int _result_rows = 0;
            int _result_cols = 5;

            float _shift_scale = 0.0f;
    };

    int MODetector::init(const char* model_path, float shift_scale, int debug) {
        // load model data
        _model_data      = load_model(model_path, &_model_data_size);
        // init rknn
        int ret                            = rknn_init(&_ctx, _model_data, _model_data_size, 0, NULL);
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return ret;
        }

        // get io num
        ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &_io_num, sizeof(_io_num));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return ret;
        }
        if (debug)             
            printf("model input num: %d, output num: %d\n", _io_num.n_input, _io_num.n_output);

        if(debug) {           
            rknn_sdk_version version;
            ret = rknn_query(_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return ret;
            }
            printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
        }

        rknn_tensor_attr input_attrs[_io_num.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < _io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret                  = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return ret;
            }
            if (debug)
            dump_tensor_attr(&(input_attrs[i]));
        }

        // get model input size
        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            if(debug)
            printf("model is NCHW input fmt\n");
            _model_input_channels = input_attrs[0].dims[1];
            _model_input_height  = input_attrs[0].dims[2];
            _model_input_width   = input_attrs[0].dims[3];
        } else {
            if(debug)
                printf("model is NHWC input fmt\n");
            _model_input_height  = input_attrs[0].dims[1];
            _model_input_width   = input_attrs[0].dims[2];
            _model_input_channels = input_attrs[0].dims[3];
        }

        if(debug)
            printf("model input height=%d, width=%d, channel=%d\n", _model_input_height, _model_input_width, _model_input_channels);

        rknn_tensor_attr output_attrs[_io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < _io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret                   = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if(debug)
                dump_tensor_attr(&(output_attrs[i]));
        } 
        // prepare model input data size
        // pre_image = cv::Mat::zeros(model_input_height, model_input_width, CV_8UC3);
        int ndims = 0;
        int total_length = 1;
        ndims = output_attrs[0].n_dims;
        for (int i = 0; i < ndims; i++) {
            total_length *= output_attrs[0].dims[i];
        }
        _result_rows = total_length / _result_cols;
        if (debug) 
            printf("model output height=%d, width=%d\n", _result_rows, _result_cols);
        
        _shift_scale = shift_scale;
        printf("MOD shift scale:%f\n", _shift_scale);
        return ret;
    }

    MODetector::~MODetector() {
        // free model_data
        if (_model_data) {
            free(_model_data);
        }

          // free ctx resource
        int ret = rknn_destroy(_ctx);
        if (ret < 0) {
            printf("rknn destroy failed.\n");
        }
    }

    // resize image and to pad image
    static cv::Mat resize_and_pad_image(cv::Mat origin_mat, float scale, int model_input_width, 
                        int model_input_height, int& pad_x0, int& pad_y0, int& pad_x1, int& pad_y1)
    {
        int image_width = origin_mat.cols;
        int image_height = origin_mat.rows;
        // resize image
        cv::Mat resized_mat;
        cv::resize(origin_mat, resized_mat, cv::Size(int(image_width * scale), int(image_height * scale)));

        // padding a image
        int h = resized_mat.rows;
        int w = resized_mat.cols;
        pad_x0 = (model_input_width - w) / 2;
        pad_y0 = (model_input_height - h) / 2;
        pad_x1 = model_input_width - w - pad_x0;
        pad_y1 = model_input_height - h - pad_y0;
        cv::Mat padded_image;
        cv::copyMakeBorder(resized_mat, padded_image, pad_y0, pad_y1, pad_x0, pad_x1, cv::BORDER_CONSTANT);
        return padded_image;
    }

    static float clip(float value, float min_val, float max_val) {
        return std::min(std::max(value, min_val), max_val);
    }

     detect_result_group MODetector::detect(const char* image_data, 
                                int image_width, int image_height, int pass_through) 
    {

        // prepare input data
        cv::Mat origin_mat(image_height, image_width, CV_8UC3, (unsigned char*)image_data);

        // get resize scale value
        float scale = std::min(_model_input_height * 1.0 / image_height, _model_input_width * 1.0 / image_width);

        int pad_x0, pad_y0, pad_x1, pad_y1;
        cv::Mat padded_image = resize_and_pad_image(origin_mat, scale, _model_input_width, _model_input_height,
                                                    pad_x0, pad_y0, pad_x1, pad_y1);

        cv::Mat gray_image;
        cv::cvtColor(padded_image, gray_image, cv::COLOR_BGR2GRAY);

        if (pass_through > 0) {
            // for debug, read the 3 channels image to input
            _pre_image = padded_image.clone();
        }
        else {
            if (_pre_image.empty()) {
                _pre_image = cv::Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC3);
                cv::Mat image_channels_data[3];
                cv::split(_pre_image, image_channels_data);
                for (int i = 0; i < 3; ++i)
                {
                    gray_image.convertTo(image_channels_data[i], CV_8U);
                }
                // merge
                cv::merge(image_channels_data, 3, _pre_image);
            }
            else {
                cv::Mat image_channels_data[3];
                cv::split(_pre_image, image_channels_data);
                image_channels_data[0] = image_channels_data[1];
                image_channels_data[1] = image_channels_data[2];
                gray_image.convertTo(image_channels_data[2], CV_8U);
                cv::merge(image_channels_data, 3, _pre_image);
            }
        }

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index        = 0;
        inputs[0].type         = RKNN_TENSOR_UINT8;
        inputs[0].size         = _model_input_width * _model_input_height * _model_input_channels;
        inputs[0].fmt          = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;
        inputs[0].buf          = (void*)_pre_image.data;

        // input data to model
        rknn_inputs_set(_ctx, _io_num.n_input, inputs);

        // get outputs
        rknn_output outputs[_io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < _io_num.n_output; i++) {
            outputs[i].want_float = 1;
        }

        int ret = rknn_run(_ctx, NULL);
        ret = rknn_outputs_get(_ctx, _io_num.n_output, outputs, NULL);

        // decode results and collect it 
        // printf("output[0] tensor size:%d", outputs[0].size);
        _inner_results.clear();
        for (int i = 0; i < _result_rows; ++i) {
            float* buf = ((float*)(outputs[0].buf)) + i * _result_cols;
            float x0 = buf[0] + _shift_scale;
            float y0 = buf[1] + _shift_scale;
            float x1 = buf[2] + _shift_scale;
            float y1 = buf[3] + _shift_scale;
            float score = buf[4];

            // only collect score greater than score_thresh
            if (score > _score_thresh) {
                x0 = (x0 * _down_scale - pad_x0) / scale;
                y0 = (y0 * _down_scale - pad_y0) / scale;
                x1 = (x1 * _down_scale - pad_x1) / scale;
                y1 = (y1 * _down_scale - pad_y1) / scale;

                // expand a little bit
                x0 = x0 - 5;
                y0 = y0 - 5;
                x1 = x1 + 5;
                y1 = y1 + 5;

                x0 = clip(x0, 0, image_width - 1);
                y0 = clip(y0, 0, image_height - 1);
                x1 = clip(x1, 0, image_width - 1);
                y1 = clip(y1, 0, image_height - 1);

                detect_result one_result;
                one_result.box.x = int(x0);
                one_result.box.y = int(y0);
                one_result.box.width = int(x1 - x0);
                one_result.box.height = int(y1 - y0);
                one_result.score = score;
                if (one_result.box.width > 0 && one_result.box.height > 0)
                    _inner_results.push_back(one_result);
            }
        }

        // release outputs every time
        ret = rknn_outputs_release(_ctx, _io_num.n_output, outputs);

        // return results
        detect_result_group result_group;
        result_group.count = _inner_results.size();
        result_group.results = (detect_result*)_inner_results.data();

        return result_group;
    }

    void MODetector::reset() 
    {
        _pre_image.release();
        // printf("image is empty:%d", _pre_image.empty());
    }
}

int mod_init(const char* model_path, moving_object_detector_context** context)
{
    seeta::MODetector* detector = new seeta::MODetector();
    int ret = detector->init(model_path, 0.5, 0);
    if (ret < 0) return ret;
    *context = (moving_object_detector_context*)detector;
    return 0;
}

int mod_init(const char* model_path, float shift_scale, moving_object_detector_context** context)
{
    seeta::MODetector* detector = new seeta::MODetector();
    int ret = detector->init(model_path, shift_scale, 0);
    if (ret < 0) return ret;
    *context = (moving_object_detector_context*)detector;
    return 0;
}

int mod_set_score_thresh(moving_object_detector_context* context, float thresh) 
{
    seeta::MODetector* detector = (seeta::MODetector*)context;
    detector->set_score_thresh(thresh);
    return 0;
}

int mod_get_score_thresh(moving_object_detector_context* context,float* thresh)
{
    seeta::MODetector* detector = (seeta::MODetector*)context;
    *thresh = detector->get_score_thresh();
    return 0;
}

int mod_detect(moving_object_detector_context* context, const char* image_data, int image_width, 
                int image_height, detect_result_group* result_group, int pass_through)
{
    seeta::MODetector* detector = (seeta::MODetector*)context;
    *result_group = detector->detect(image_data, image_width, image_height, pass_through);
    return 0;
}

int mod_deinit(moving_object_detector_context* context)
{   
    seeta::MODetector* detector = (seeta::MODetector*)context;
    if (detector)
        delete detector;

    return 0;
}

int mod_reset(moving_object_detector_context* context)
{
    seeta::MODetector* detector = (seeta::MODetector*)context;
    detector->reset();
    return 0;
}