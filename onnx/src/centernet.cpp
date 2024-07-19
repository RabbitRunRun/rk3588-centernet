#include <centernet.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/dnn.hpp>
#include <time.h>

CenterNet::CenterNet(const char * model_path)
{
    net = cv::dnn::readNet(model_path);
    std::cout << "model loaded" << std::endl;
    input_size = 512;
    pre_image = cv::Mat::zeros(input_size, input_size, CV_32FC3);
    //model_fps = 0;
}
bool CenterNet::load_model(const char *  model_path)
{
    net = cv::dnn::readNet(model_path);
    std::cout << "model loaded" << std::endl;
    input_size = 512;
    pre_image = cv::Mat::zeros(input_size, input_size, CV_32FC3);
    //model_fps = 0;
    return true;
}

std::vector<std::vector<float>> CenterNet::process(cv::Mat image)
{
    // resize image
    
    int h = image.rows;
    int w = image.cols;
    double scale = std::min(input_size * 1.0 / h, input_size * 1.0 / w);
    cv::Mat new_image;
    cv::resize(image, new_image, cv::Size(int(w*scale), int(h*scale)));

    int h_ = new_image.rows;
    int w_ = new_image.cols;
    int pad_x0 = (input_size - w_) / 2;
    int pad_y0 = (input_size - h_) / 2;
    int pad_x1 = input_size - w_ - pad_x0;
    int pad_y1 = input_size - h_ - pad_y0;

    cv::Mat padded_image;
    cv::copyMakeBorder(new_image, padded_image, pad_y0, pad_y1, pad_x0, pad_x1, cv::BORDER_CONSTANT);

    cv::Mat gray_image;
    cv::cvtColor(padded_image, gray_image, cv::COLOR_BGR2GRAY);

    if (pre_image.empty())
    {
        pre_image = cv::Mat::zeros(gray_image.rows, gray_image.cols, CV_32FC3);
        for (int i = 0; i < 3; ++i)
        {
            gray_image.convertTo(pre_image.col(i), CV_32F);
        }
    }
    else
    {
        cv::Mat channels[3];
        cv::split(pre_image, channels);
        channels[0] = channels[1].clone();
        channels[1] = channels[2].clone();
        gray_image.convertTo(channels[2], CV_32F);
        cv::merge(channels, 3, pre_image);
    }

    
    cv::Mat blob = cv::dnn::blobFromImage(pre_image, 1.0, cv::Size(input_size, input_size), cv::Scalar(0, 0, 0));
    net.setInput(blob);
    cv::Mat output = net.forward();
    cv::Mat det_output(output.size[1], output.size[2], CV_32F, output.ptr<float>());

   

    std::vector<std::vector<float>>result;
    double down_scale = 7.6418;
    for (int i = 0; i < det_output.rows; ++i)
    {
        cv::Vec<float, 5> bnd = det_output.at<cv::Vec<float, 5>>(i);
        float x0 = (bnd[0] * down_scale - pad_x0) / scale;
        float y0 = (bnd[1] * down_scale - pad_y0) / scale;
        float x1 = (bnd[2] * down_scale - pad_x0) / scale;
        float y1 = (bnd[3] * down_scale - pad_y0) / scale;
        float score = bnd[4];
        if (score > 0.5)
        {
            std::vector<float> res;
            res.push_back(x0);
            res.push_back(y0);
            res.push_back(x1);
            res.push_back(y1);
            res.push_back(score);
            result.push_back(res);
        }
    }
    
    // std::cout<<model_fps<<std::endl;
    // 处理结果
    // ...
    return result;
}
