#ifndef MOVING_OBJECT_DETECTOR_H_
#define MOVING_OBJECT_DETECTOR_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <stdint.h>

#define API_EXPORT __attribute__((visibility("default")))

struct moving_object_detector_context;

// (x,y) means left top position
struct bbox {
    int x;
    int y;
    int width;
    int height;
};

struct detect_result {
    bbox box;
    float score;
};

struct detect_result_group {
    int count;
    detect_result* results;
};

/*
method:init
inputs:
    model_path: rknn model to init from
    context: to get detector context from this method.Its type is moving_object_detector_context**.
outputs:
    0 means succeed. otherwise failed.
*/
API_EXPORT int mod_init(const char* model_path, moving_object_detector_context** context);

/*
method:init
inputs:
    model_path: rknn model to init from
    shift_scale: shift scale for detector results box. Default is 0.5, it's valued from [0.0-1.0].
    context: to get detector context from this method.Its type is moving_object_detector_context**.
outputs:
    0 means succeed. otherwise failed.
*/
API_EXPORT int mod_init(const char* model_path, float shift_scale, moving_object_detector_context** context);


/*
method:set_score_thresh
inputs:
    context: detector's context, this value is derived from calling init method
    thresh: set score thresh to filter detection results.Only detection results which confidence score are 
    great than thresh can be remained and returned in method detect
outputs:
    0 means succeed. otherwise failed.
*/
API_EXPORT int mod_set_score_thresh(moving_object_detector_context* context, float thresh);

/*
method:get_score_thresh
inputs:
    context: detector's context, this value is derived from calling init method
    thresh: get score thresh.Its type is float*
outputs:
    0 means succeed. otherwise failed.
*/
API_EXPORT int mod_get_score_thresh(moving_object_detector_context* context,float* thresh);

/*
method:detect
inputs:
    context: detector's context, this value is derived from calling init method
    image_data: image data, its format is bgr888, equal to opencv cv::Mat format.
    image_width: image's width
    image_height: image's height
    detect_result_group: returned moving object results detected. Its type is detect_result_group*
    pass_through:pass_through=1 only used for debugging. Please set 0 to it when using!!!
outputs:
    0 means succeed. otherwise failed.
*/

API_EXPORT int mod_detect(moving_object_detector_context* context, const char* image_data, int image_width, 
                int image_height, detect_result_group* result_group, int pass_through);

/*
method:reset
inputs:
    context: detector's context, this value is derived from calling init method.Must call it when
    long time passed after last calling mod_detect method.

outputs:
    0 means succeed. otherwise failed.
*/

API_EXPORT int mod_reset(moving_object_detector_context* context);


/*
method:deinit to free resources allocated
inputs:
    context: detector's context, this value is derived from calling init method
outputs:
    0 means succeed, otherwise means error occured.
*/
API_EXPORT int mod_deinit(moving_object_detector_context* context);

#ifdef _cplusplus
}
#endif

#endif // MOVING_OBJECT_DETECTOR_H_