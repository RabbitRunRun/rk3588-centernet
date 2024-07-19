#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "moving_object_detector.h"

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int main_single_image(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: main model_path image_path.\n");
        exit(-1);
    }
    // init model
    const char* model_path = argv[1];
    // read a image and to detect
    const char* image_path = argv[2];
    cv::Mat image = cv::imread(image_path);
    printf("image width:%d height:%d channels:%d", image.cols, image.rows, image.channels());

    moving_object_detector_context* context = NULL;
    int ret = mod_init(model_path, &context);
    if(ret < 0) {
        printf("Model init failed.");
        return -1;
    }

    //set score thresh
    float score_thresh = 0.5f;
    if (argc >=4) {
        score_thresh = atof(argv[3]);
    }

    ret = mod_set_score_thresh(context, score_thresh);
    if(ret < 0) {
        printf("Set score thresh failed.");
        return -1;
    }

    // get score thresh and print
    float thresh;
    ret = mod_get_score_thresh(context, &thresh);
    if(ret < 0) {
        printf("Get score thresh failed.");
        return -1;
    }
    printf("Score thresh:%f\n", thresh);

    detect_result_group result_group;
    ret = mod_detect(context, (const char*)image.data, image.cols, 
                image.rows, &result_group, 1);
    printf("Detected %d objects.\n", result_group.count);

    int test_count = 20;
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i) 
    {
        ret = mod_detect(context, (const char*)image.data, image.cols, 
                    image.rows, &result_group, 1);
    }
    gettimeofday(&stop_time, NULL);
    printf("Detector once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);
    // to draw results on the image
    for (int i = 0; i < result_group.count; ++i) {
        float score = result_group.results[i].score;
        printf("score:%f", score);
        bbox box = result_group.results[i].box;
        printf("box location:[x,y,width,height]=[%d, %d, %d, %d]\n", box.x, box.y, box.width, box.height);
        cv::rectangle(image, cv::Rect(box.x, box.y, box.width, box.height), CV_RGB(255, 0, 0), 3);
        cv::putText(image, std::to_string(score), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
    }

    // cv::imshow("result", image);
    // cv::waitKey(1000);
    // save result png pic
    int pos = std::string(image_path).find_last_of(".");
    std::string pure_output_name = std::string(image_path).substr(0, pos);
    printf("output image name:result_%s.jpg\n", pure_output_name.c_str());
    cv::imwrite(std::string("/tmp/result_") + pure_output_name + std::string(".jpg"), image);

    return 0;
}


int main_3_serial_images(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: main model_path image_path1 image_path2 image_path3.\n");
        exit(-1);
    }
    // init model
    const char* model_path = argv[1];
    // read a image and to detect
    const char* image_path1 = argv[2];
    const char* image_path2 = argv[3];
    const char* image_path3 = argv[4];
    cv::Mat image1 = cv::imread(image_path1);
    cv::Mat image2 = cv::imread(image_path2);
    cv::Mat image3 = cv::imread(image_path3);
    printf("image width:%d height:%d channels:%d\n", image1.cols, image1.rows, image1.channels());

    moving_object_detector_context* context = NULL;
    int ret = mod_init(model_path, 0.5f, &context);
    if(ret < 0) {
        printf("Model init failed.");
        return -1;
    }

    //set score thresh
    float score_thresh = 0.5f;
    if (argc >=6) {
        score_thresh = atof(argv[5]);
    }

    ret = mod_set_score_thresh(context, score_thresh);
    if(ret < 0) {
        printf("Set score thresh failed.");
        return -1;
    }

    // get score thresh and print
    float thresh;
    ret = mod_get_score_thresh(context, &thresh);
    if(ret < 0) {
        printf("Get score thresh failed.");
        return -1;
    }
    printf("Score thresh:%f\n", thresh);

    detect_result_group result_group;


    int test_count = 20;
    if (argc >= 7) {
        test_count = atoi(argv[6]);
        printf("test times:%d", test_count);
    }

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i) 
    {

        ret = mod_detect(context, (const char*)image1.data, image1.cols, 
                    image1.rows, &result_group, 0);
        // printf("Detected %d objects im image1.\n", result_group.count);

        ret = mod_detect(context, (const char*)image2.data, image2.cols, 
                    image2.rows, &result_group, 0);
        // printf("Detected %d objects im image2.\n", result_group.count);

        ret = mod_detect(context, (const char*)image3.data, image3.cols, 
                    image3.rows, &result_group, 0);
        // printf("Detected %d objects im image3.\n", result_group.count);
    }
    gettimeofday(&stop_time, NULL);
    printf("Detector once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count / 3.0);

    // to draw results on the image
    for (int i = 0; i < result_group.count; ++i) {
        float score = result_group.results[i].score;
        printf("score:%f", score);
        bbox box = result_group.results[i].box;
        printf("box location:[x,y,width,height]=[%d, %d, %d, %d]\n", box.x, box.y, box.width, box.height);
        cv::rectangle(image3, cv::Rect(box.x, box.y, box.width, box.height), CV_RGB(255, 0, 0), 3);
        cv::putText(image3, std::to_string(score), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
    }

    int pos = std::string(image_path3).find_last_of(".");
    int pos_ = std::string(image_path3).find_last_of("/");
    std::string pure_output_name;
    if (pos_ >= 0) {
        pure_output_name  = std::string(image_path3).substr(pos_ + 1, pos - pos_ -1);
    }
    else {
        pure_output_name  = std::string(image_path3).substr(0, pos);
    }
    printf("output image name:result_%s.jpg\n", pure_output_name.c_str());
    cv::imwrite(std::string("/tmp/result_") + pure_output_name + std::string(".jpg"), image3);

    mod_reset(context);
    return 0;
}

int main(int argc, char** argv) {
    return main_3_serial_images(argc, argv);
}
