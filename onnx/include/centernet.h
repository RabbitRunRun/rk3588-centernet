# include <opencv2/opencv.hpp>
# include <opencv2/dnn.hpp>
# include <vector>
# include <iostream>


class CenterNet{
private:
    cv::dnn::Net net;
    int input_size;
    cv::Mat pre_image;
public:
    CenterNet(const char * model_path);
    bool load_model(const char * model_path);
    std::vector<std::vector<float>> process(cv::Mat image);
    
    cv::Mat vis_result(cv::Mat image, std::vector<std::vector<float>> result){
        for(int i = 0; i < result.size(); i++){
            int x0 = result[i][0];
            int y0 = result[i][1];
            int x1 = result[i][2];
            int y1 = result[i][3];
            int score = int(result[i][4] * 100);
            cv::putText(image, std::to_string(score), cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            //cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2);
            }
        return image;
    };

};