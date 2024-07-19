#include<iostream>
#include<stdlib.h>
#include<centernet.h>
#include<opencv2/opencv.hpp>
#include<time.h>

int main(){
    std::cout<<"init"<<std::endl;

    // open video
    cv::VideoCapture cap("demo_big.mp4");
    if(!cap.isOpened()){
        std::cout<<"open video failed"<<std::endl;
        return -1;
    }

    // init CenterNet
    CenterNet cnet = CenterNet("center_net_lite.onnx");

    // process video
    cv::Mat frame;
    while(cap.read(frame)){
        
        // process frame
        std::vector<std::vector<float>> objs;
        double start_time = std::clock();
        objs = cnet.process(frame);
        double end_time = std::clock();
        std::cout<<"time: "<<1.0/((end_time - start_time)/CLOCKS_PER_SEC)<<std::endl;
        // draw bbox
        frame = cnet.vis_result(frame, objs);
        cv::imshow("frame", frame);
        cv::waitKey(10);
    }
    return 0;
}
