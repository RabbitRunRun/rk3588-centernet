#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<algorithm>
#ifndef _Img_Preprocess
#define _Img_Preprocess

class Preprocess{
    public:
    // fix image (h, w, c) to (size, size, c)
    cv::Mat pad_resize(cv::Mat img, int size){
        float scale = 0;
        int px = 0;
        int py = 0;
        int h = img.rows;
        int w = img.cols;
        if( w > h){
            scale = size * 1.0/w;
            h = int(h * scale);
            w = size;
        }else{
            scale = size * 1.0/h;
            w = int(w * scale);
            h = size;
        }
        cv::Mat new_img;
        cv::resize(img, new_img, cv::Size(w, h));
        px = (size - w)/2;
        py = (size - h)/2;
        cv::copyMakeBorder(new_img, new_img, py, std::max(size-h-py, 0), px, std::max(size-w-px, 0), cv::BORDER_CONSTANT, 0);
        return new_img;
    };

    cv::Mat pad_resize(cv::Mat img, int size, float & scale, int & px, int & py){
        int h = img.rows;
        int w = img.cols;
        if( w > h){
            scale = size * 1.0/w;
            h = int(h * scale);
            w = size;
        }else{
            scale = size * 1.0/h;
            w = int(w * scale);
            h = size;
        }
        cv::Mat new_img;
        cv::resize(img, new_img, cv::Size(w, h));
        px = (size - w)/2;
        py = (size - h)/2;
        cv::copyMakeBorder(new_img, new_img, py, std::max(size-h-py, 0), px, std::max(size-w-px, 0), cv::BORDER_CONSTANT, 0);
        return new_img;
    };

    cv::Mat to_tensor(cv::Mat img){
         int w = img.rows;
        int h = img.cols;
        img.convertTo(img, CV_32FC3);
        cv::Mat blob = cv::dnn::blobFromImage(img);
        return blob;
    };
};
#endif