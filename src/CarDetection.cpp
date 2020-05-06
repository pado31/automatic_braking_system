#include "CarDetection.hpp"
 
using namespace std;
using namespace cv;

/**
	Constructor of the class.

	@param image = image useful to print the final result.
	@param segmented = image in which is defined the ROI where detect the car.
	@param min_y = upper limit of the road detected
	@param max_y = lower limit of the road detected
*/
CarDetection::CarDetection(Mat image, Mat segmented, int min_y, int max_y){
	this -> image = image;
	this -> segmented = segmented;
	this -> min_y = min_y;
	this -> max_y = max_y;
}

/**
	Method that blurs the image and using Canny detects the edges.
*/
void CarDetection::segmentation(){

	// Convert the image with the ROI to gray.
	Mat gray;
	cvtColor(this -> segmented, gray, CV_BGR2GRAY);

	blur(gray, gray, Size(5,5));

	Mat detected_edges;
	Canny(gray, detected_edges, 30, 90);

	this -> cannyResult = Scalar::all(0);
    this -> image.copyTo( this -> cannyResult, detected_edges); //Save the result

}

/**
	Build the summed area table of the input image. The result in saved in cv::Mat sat.
*/
void CarDetection::summedAreaTable(Mat input){

	this -> sat = Mat(this -> cannyResult.size(), CV_64F);
	
	int pixel_value = 0; //Zero if the pixel is balck, one otherwise.
	
	for (int x = 0; x < this -> cannyResult.cols; ++x){
		for (int y = 0; y < this -> cannyResult.rows; ++y){
			
			if (this -> cannyResult.at<Vec3b>(Point(x,y)) == black){
				pixel_value = 0;
			}
			else{
				pixel_value = 1;
			}
			
			if(x == 0 && y == 0){
				this -> sat.at<float>(Point(x,y)) = pixel_value;
			}
			else if(x == 0){
				this -> sat.at<float>(Point(x,y)) = pixel_value + this -> sat.at<float>(Point(x,y-1)); 
			}
			else if(y == 0){
				this -> sat.at<float>(Point(x,y)) = pixel_value + this -> sat.at<float>(Point(x-1,y)); 
			}
			else{
				this -> sat.at<float>(Point(x,y)) = pixel_value + this -> sat.at<float>(Point(x-1,y)) + 
					this -> sat.at<float>(Point(x,y-1)) - this -> sat.at<float>(Point(x-1, y-1));
			}
			
		}
	}
}

/**
	Given a summed area table, find the area of the image with the maximum edge density.

	@param summedAreaTable = input summed area table.
*/
void CarDetection::findOptDensity(Mat summedAreaTable){
	this -> message = 0; //None obstacle.

	Point topLeft_corner;
	topLeft_corner.x = 0;
	topLeft_corner.y = 0;
	int window_size = image.cols / 2; //Initial window size.
	int x_limit = image.cols;
	int y_limit = max_y + 200; //bottom limit of the road plus an offset 

	float maxDensity; //prov max density of the current window
	
	Point corner_prov;
	corner_prov.x = 0;
	corner_prov.y = 0;

	bool stop = false;

	while(!stop && window_size > min_window_size){
		
		maxDensity = 0;
		
		for (int x = topLeft_corner.x; x + window_size < x_limit; ++x){
			for (int y = topLeft_corner.y; y + window_size < y_limit; ++y){
				
				//Number pixel edges in the area.
				int whitePixels = summedAreaTable.at<float>(Point(x,y)) + summedAreaTable.at<float>(Point(x+window_size,y+window_size))
					- summedAreaTable.at<float>(Point(x+window_size,y)) - summedAreaTable.at<float>(Point(x,y+window_size));
				float density = whitePixels / (float)(window_size*window_size);

				if (density > maxDensity){
					corner_prov.x = x;
					corner_prov.y = y;
					maxDensity = density;
				}

			}
		}
		
		//Update
		topLeft_corner.x = corner_prov.x;
		topLeft_corner.y = corner_prov.y;
		window_size =  window_size - 10;
		
		
		if (maxDensity > 0.065){
			line(image, topLeft_corner, Point(topLeft_corner.x + window_size, topLeft_corner.y), Scalar(0,0,255), 15, 8);
			line(image, topLeft_corner, Point(topLeft_corner.x, topLeft_corner.y + window_size), Scalar(0,0,255), 15, 8);
			line(image, Point(topLeft_corner.x + window_size, topLeft_corner.y + window_size), Point(topLeft_corner.x, topLeft_corner.y + window_size), Scalar(0,0,255), 15, 8);
			line(image, Point(topLeft_corner.x + window_size, topLeft_corner.y), Point(topLeft_corner.x + window_size, topLeft_corner.y + window_size), Scalar(0,0,255), 15, 8);
			stop = true;

			getPriority(topLeft_corner, window_size);
		}
		
		
	}

	cout << "Window size " << window_size << endl;

}

/**
	Method that computes if the car in front is too close.

	@param topleft = coordinates of the top left corner of the window.
	@param window_size = size of the window
	@return int = 2 if the car is too close, esle 1.
*/
int CarDetection::getPriority(Point topleft, int window_size){
	Point car;
	car.x = topleft.x + window_size;
	car.y = topleft.y + window_size;

	float center = min_y + (max_y - min_y)*0.6;

	if(car.y < center){
		this -> message = 1;
	}
	else{
		this -> message = 2;
	}

}

/**
	 Public method that, using all the private methods, executes all the procedures to find
   	 the car.
*/
void CarDetection::detectCar(){

	CarDetection::segmentation();

	CarDetection::summedAreaTable(cannyResult);

	CarDetection::findOptDensity(sat);
}

/**
    @return Mat = image with the detected car.
*/
Mat CarDetection::getDetectedCar(){
	return image;
}

/**
    @return int = level of alert if the car in front is too close.
*/
int CarDetection::getMessage(){
	return message;
}
