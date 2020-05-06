#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

class LaneLines
{
	private:
    	cv::Mat image;
    	cv::Mat laneImage;
    	cv::Mat final;
    	cv::Mat regionImage;

    	std::vector<cv::Vec4i> lines;

    	std::vector<cv::Vec4i> finalPoints;
    	float m1;
    	float q1;
    	float m2;
    	float q2;

    	int max_y;
    	int min_y;

    	cv::Vec3b black = cv::Vec3b(0,0,0);
    	cv::Vec3b blue = cv::Vec3b(255,0,0);


	public:
		LaneLines(cv::Mat);
		void processRoad();
		cv::Mat getRecognizedLines();
		cv::Mat getRegionImage();
		int getMaxY();
		int getMinY();


	private:
		void selectColor();
		cv::Mat setRegionOfInterest(cv::Mat);
		cv::Mat edgeDetector(cv::Mat);
		void defineLaneLines(std::vector<cv::Vec4i>);
		std::vector<float> selectSlopeCoefficients(std::vector<cv::Vec4i>);
		void color();
		void createRegion();
		void findLineParams(std::vector<cv::Vec4i>);


 
};