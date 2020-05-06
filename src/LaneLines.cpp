#include "LaneLines.hpp"

using namespace std;
using namespace cv;

/**
    Constructor of the class.

    @param image =  image to be processed
*/
LaneLines::LaneLines(cv::Mat image){
    
    this -> image = image;
}

/**
    Method that filters the image keeping only the white and yellow pixels. The result is saved
    in the variable cv::laneImage. 
*/
void LaneLines::selectColor(){

    //Image with same size of the original image that will contain only the interested colors
    this -> laneImage = Mat(image.rows, image.cols, CV_8UC1, Scalar(0,0,0));


	Mat hls_image; 
	cvtColor(image, hls_image, COLOR_BGR2HLS);

	Mat whiteMask;
	inRange(hls_image, Scalar(0,190,0), Scalar(255,255,255), whiteMask);
	Mat yellowMask;
	inRange(hls_image, Scalar(20,120,100), Scalar(40,200,255), yellowMask);

	Mat mask; 
	bitwise_or(whiteMask, yellowMask, mask); //merge masks

	bitwise_and(image, image, laneImage, mask=mask); //merge mask and original image

}

/**
    Method that defines the region of interest of the image. All the pixels that do not
    belong to the ROI are set to black. The region of interest is a triangle which verteces 
    are proportional to image dimensions and they cover the area of the image in which 
    very likely there is the road.
    
    @param input = image to which set the ROI.
    @return Mat = input image converted to gray color-space with no-interesting pixels 
                  set to black.
*/
cv::Mat LaneLines::setRegionOfInterest(cv::Mat input){
	
	Mat out; //output image

	Point b_left, b_right, center; //vertices of the triangle
	
    Size s = input.size();

	b_left.x = cvRound(s.width*0.25);
	b_left.y = cvRound(s.height*0.66);

	b_right.x = cvRound(s.width*0.85);
	b_right.y = cvRound(s.height*0.66);

	center.x = cvRound(s.width*0.5);
	center.y = cvRound(s.height*0.5);


    // Slopes (m) and constant (q) of the line equation in the form y = mx + q

	float mLeft = (b_left.y - center.y) / (float)(b_left.x - center.x);
    float qLeft = b_left.y - mLeft * b_left.x;

    float mRight = (b_right.y - center.y) / (float)(b_right.x - center.x);
    float qRight = b_right.y - mRight * b_right.x;

    for (int y = 0; y < s.height; y++) {
            for (int x = 0; x < s.width; x++) {
                // Color pixel only if is below the two lines
                if (!((y > mLeft * x + qLeft) && (y > mRight * x + qRight))) {
                    input.at<Vec3b>(Point(x,y)) = black;
                    
                }
            
            }
    }


    cvtColor(input, out, CV_BGR2GRAY);

    return out;
}

/**
    Method that detects edges in the input image.

    @param input = gray image to be processed.
    @return Mat = gray image with the detected edges. 
*/
cv::Mat LaneLines::edgeDetector(cv::Mat input){

    GaussianBlur(input, input, Size(15,15), 0); //blurs the input image

    Mat detected_edges;
    Canny(input, detected_edges, 60, 180); //Canny edge detector

    return detected_edges;
}

void LaneLines::defineLaneLines(std::vector<cv::Vec4i> input){

	// Select the proper splope coefficient
    std::vector<float> l_r_slopes = selectSlopeCoefficients(input);
	float min_left =  l_r_slopes[0];
	float max_right =  l_r_slopes[1];
	
    std::vector<float> weights; //vector that contains the lenght of each line
	std::vector<float> slopes; //vector that contains the slope coefficient of each line 
	std::vector<cv::Vec4i> selected_lines; //vector that contains the lines that satisfy some requirements

	// process the entire input
    for( size_t i = 0; i < input.size(); i++ ){

        //slope coefficient of the current line
		float m = (input[i][1] - input[i][3]) / (float)(input[i][0] - (input[i][2]));
        
        //keep the line only if it is not (almost) orizontal and only if the slope is similar to the proper one.
		if(!(m < 0.05 && m > -0.05) && (m < min_left + 0.02 || m > max_right - 0.02) && (m > -1 && m < 1)){
			
            // Add length of the current line to the vector
            weights.push_back(sqrt(((input[i][1] - input[i][3])*(input[i][1] 
        	   - input[i][3])) + ((input[i][0] - input[i][2])*(input[i][0] - input[i][2]))));
        	
            slopes.push_back(m); // Add slope of the current line to the vector

        	
        	Vec4i selected_points; //points that identify the current line
        	selected_points[0] = input[i][0];
        	selected_points[1] = input[i][1];
        	selected_points[2] = input[i][2];
        	selected_points[3] = input[i][3];
        	selected_lines.push_back(selected_points); //Add to the vector of the interesting lines.
		}
    }

    /*
        After selecting the lines with a correct slope coefficient we need to determine the left 
        and the right lines of the road. In order to do that we compute a mean of the left and 
        right lines belonging to the vector selected_lines. The mean is computed on the
        length. It is used a weighted average in order to give more importante to longer (stronger) 
        lines.
    */    


    // The two points that identify the left lane line
    float pt1_x = 0;
    float pt1_y = 0;
    float pt2_x = 0;
    float pt2_y = 0;
    // The two points that identify the right lane line
    float pt3_x = 0;
    float pt3_y = 0;
    float pt4_x = 0;
    float pt4_y = 0;

    // Sum of the left and right line lengths.
    float left_weight = 0;
    float right_weight = 0;

    for( size_t i = 0; i < selected_lines.size(); i++ ){
        
        if (slopes[i] < 0){ //left lines
        	left_weight = left_weight + weights[i];

        	pt1_x = pt1_x + selected_lines[i][0] * weights[i];
        	pt1_y = pt1_y + selected_lines[i][1] * weights[i];
        	pt2_x = pt2_x + selected_lines[i][2] * weights[i];
        	pt2_y = pt2_y + selected_lines[i][3] * weights[i];
        }
        else{ //right lines
        	right_weight = right_weight + weights[i];

        	pt3_x = pt3_x + selected_lines[i][0] * weights[i];
        	pt3_y = pt3_y + selected_lines[i][1] * weights[i];
        	pt4_x = pt4_x + selected_lines[i][2] * weights[i];
        	pt4_y = pt4_y + selected_lines[i][3] * weights[i];
        }
    }

    Vec4i firstLine;
	firstLine[0] = pt1_x / left_weight;
    firstLine[1] = pt1_y / left_weight;
    firstLine[2] = pt2_x / left_weight;
    firstLine[3] = pt2_y / left_weight;

    finalPoints.push_back(firstLine);


    Vec4i secondLine;
	secondLine[0] = pt3_x / right_weight;
    secondLine[1] = pt3_y / right_weight;
    secondLine[2] = pt4_x / right_weight;
    secondLine[3] = pt4_y / right_weight;
	
	finalPoints.push_back(secondLine);

	LaneLines::findLineParams(finalPoints); //save lines coefficients for future use
}


/**
    This method finds the minimum slope coefficient for the left lane lies and the maximum
    slope coefficient for the right ones.

    @param input = lines to be processed.
    @return a vector with the two slope coefficients needed.
*/
std::vector<float> LaneLines::selectSlopeCoefficients(std::vector<cv::Vec4i> input){
	
    float max_right = 0;
	float min_left = 0;

	for( size_t i = 0; i < input.size(); i++ ){

		float m = (input[i][1] - input[i][3]) / (float)(input[i][0] - (input[i][2]));

		if (m < 0 && m < min_left && m > -1){
			min_left = m;
		}
		else if(m > 0 && m > max_right && m < 1){
			max_right =  m;
		}
	}

	std::vector<float> out;
	out.push_back(min_left);
	out.push_back(max_right);

	return out;
}

/**
    Method that finds the slope coefficient and the constant of two lines. 

    @param input = the vector with the points that identify the two lines.
*/
void LaneLines::findLineParams(std::vector<cv::Vec4i> input){

    //Works only if there are two lines
    if(input.size() == 2){
	    this -> m1 = (input[0][1] - input[0][3]) / (float)(input[0][0] - input[0][2]);
        this -> q1 = input[0][1] - m1 * (input[0][0]);
        this -> m2 = (input[1][1] - input[1][3]) / (float)(input[1][0] - input[1][2]);
        this -> q2 = input[1][1] - m2 * (input[1][0]);
    }

}

/**
    Method that colors the portion of road detected.
*/
void LaneLines::color(){
	
    Mat prov;
	image.copyTo(prov);

	Size s =  prov.size();

    /*
    We need to computed the top and down limits of the two lines in order to build a 
    trapeze, that is the region to color.
    */
    int max_y_left =  max(finalPoints[0][1], finalPoints[0][3]);
    int max_y_right = max(finalPoints[1][1], finalPoints[1][3]);
    int min_y_left =  min(finalPoints[0][1], finalPoints[0][3]);
    int min_y_right = min(finalPoints[1][1], finalPoints[1][3]);

    //These values will be useful for improving the car detection procedure too.
    this -> min_y = min(min_y_left, min_y_right);
    this -> max_y = max(max_y_left, max_y_right);


    for (int y = 0; y < s.height; y++) {
            for (int x = 0; x < s.width; x++) {
                // Color pixel only if is below the two lines and between min_y and max_y
                if (((y > m1 * x + q1) && (y > m2 * x + q2)) && y > min_y && y < 1900) {
                    prov.at<Vec3b>(Point(x,y)) = blue;
                }
            
        }
    }

    addWeighted(this -> image, 0.6, prov, 0.4, 0, this -> final); //blurs a bit the blue pixels
}

/**
    Based on the portion of road detected, this method computes an image in which all the pixels
    that do not belong to the portion of road in front of the car are set to black.
    This method is useful because produces an image that optimizes the detection of the car.

*/
void LaneLines::createRegion(){
    
    float prov_m1 = this -> m1; 
	float prov_m2 = this -> m2; 

	image.copyTo(this -> regionImage);
	Size s =  this -> regionImage.size();

    for (int y = 0; y < s.height; y++) {
            for (int x = 0; x < s.width; x++) {
                /*
                Color pixel only if is below the two lines. The lines are traslated of 100 px in order to properly set the ROI.
                */
                if (!((y > prov_m1 * x + q1 - 100) && (y > prov_m2 * x + q2 - 100) || y > max_y + 200 )) {
                    this -> regionImage.at<Vec3b>(Point(x,y)) = black;
                }
            
        }
    }
    
}

/**
    Public method that, using all the private methods, executes all the procedures to find
    the road.
*/
void LaneLines::processRoad(){

    LaneLines::selectColor();
    
    Mat out;

    out = LaneLines::setRegionOfInterest(this -> laneImage);

    out = LaneLines::edgeDetector(out);

    HoughLinesP(out, this -> lines, 1, 1 * CV_PI/180, 90, 30, 50 );

    LaneLines::defineLaneLines(this -> lines);

    LaneLines::color();

    LaneLines::createRegion();

}

/**
    @return Mat = image with the detected portion of road.
*/
Mat LaneLines::getRecognizedLines(){
	return final;
}

/**
    @return Mat = image with the ROI for car detection.
*/
Mat LaneLines::getRegionImage(){
	return regionImage;
}

/**
    @return int = lower limit of the road for car detection.
*/
int LaneLines::getMinY(){
	return min_y;
}

/**
    @return int = upper limit of the road for car detection.
*/
int LaneLines::getMaxY(){
	return max_y;
}