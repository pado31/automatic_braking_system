#include "LaneLines.hpp"
#include "CarDetection.hpp"


#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    //Message images   
    Mat processing = imread("../images/messages/processing.jpg");
    Mat free = imread("../images/messages/free.jpg");
    Mat attention = imread("../images/messages/attention.jpg");
    Mat slowdown = imread("../images/messages/slowdown.jpg");

    Mat dst(Size(4000, 4200), CV_64F, Scalar::all(0));

    int n_images = 12; 
    String general_path =  "../images/i";

    auto total_time = 0; //Useful to calculate average execution time

    for (int i = 1; i <= n_images; ++i){
        
        String path = general_path + to_string(i) + ".JPG";
        String window_name = "Image" + to_string(i);

        // Load image
        Mat img = imread(path);

        cout << "Image " << i << endl;
        
        vconcat(img, processing, dst);
        namedWindow(window_name, WINDOW_NORMAL);
        imshow(window_name, dst);
        waitKey(1);
        

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        LaneLines obj (img);

        obj.processRoad();

        Mat result = obj.getRegionImage();
        Mat lines = obj.getRecognizedLines();
        int min_y = obj.getMinY();
        int max_y = obj.getMaxY();
        
        vconcat(lines, processing, dst);
        namedWindow(window_name, WINDOW_NORMAL);
        imshow(window_name, dst);
        waitKey(1);
        
        CarDetection obj2 (lines, result, min_y, max_y);
        obj2.detectCar();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cout << "Duration " << duration << "ms" << endl;
        total_time = total_time + duration;


        Mat final_result = obj2.getDetectedCar();

        int message = obj2.getMessage();

        if (message == 0){
            vconcat(final_result, free, dst);
        }
        else if (message == 1){
            vconcat(final_result, attention, dst);
        }
        else if (message == 2){
            vconcat(final_result, slowdown, dst);
        }
        
        
        namedWindow(window_name, WINDOW_NORMAL);
        imshow(window_name, dst);
        waitKey(1);
        
    }

    cout << " " << endl;
    cout << "Average duration " << total_time/n_images << "ms" << endl;

    waitKey(0);

	return 0;
}
