#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <chrono>

class CarDetection
{
	private:
    	cv::Mat image;
    	cv::Mat segmented;
    	cv::Mat cannyResult;
    	cv::Mat sat;

    	int min_y;
    	int max_y;

    	cv::Vec3b black = cv::Vec3b(0,0,0);

    	int const min_window_size = 200;

    	int message;
    
    public: 
    	CarDetection(cv::Mat original, cv::Mat regionImage, int min, int max);
    	void detectCar();
    	cv::Mat getDetectedCar();
    	int getMessage();

    private:
    	void segmentation();
    	void summedAreaTable(cv::Mat);
    	void findOptDensity(cv::Mat);
    	int getPriority(cv::Point, int);

};