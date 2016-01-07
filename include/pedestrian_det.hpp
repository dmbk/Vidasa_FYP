#include<stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "cv.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "main_processor.hpp"
using namespace cv;
using namespace std;



class pedestrian_det: public abstract_det
{

    private:
        static pedestrian_det * detector_obj;

        static const int width_roi = 400;
        static const int height_roi = 200; //assumes these are lesser than the image width and height
        static int x_roi, y_roi;
        static Mat frame1;
	static int MAX_PED_DENSITY;
	static int ped_score_red_rate;

        HOGDescriptor hog;
        CascadeClassifier cascade;
        VideoCapture cap;
		int updated_count;
        Rect main_roi;
        int initialized;

        static void onMouse(int event, int x, int y, int, void *);
        pedestrian_det(int argc, char ** argv);
        int detect_hog_svm(Mat & img);
        void draw_ROI_poly(Mat & img);
        Rect get_roi(Mat & img);
        

       
        int init(int argc, char ** argv); //only supposed to be initialized via the constructor
	protected:
		int process_equation(int input_val);
    public:
        static pedestrian_det * get_detetctor(int argc, char ** argv);

        void start_session();
	int do_iteration();
	int get_updated_count()
        {
            int temp = updated_count;
            updated_count = 0;
            return temp;
        }



};



int pedestrian_det::init(int argc, char ** argv)
{
    printf("\nname =%s\n", argv[0]);
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    if(cascade.load("data/ped.xml"))
    {
        return -1;
    }

    cap = VideoCapture(argv[1]);
	
    if(!cap.isOpened())
    {
        return -1;
    }
    if(argc==3){    
	MAX_PED_DENSITY=atoi(argv[2]);
}
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    Mat image_first;
    cap >> image_first;
    main_roi = get_roi(image_first); //defined using mouse click
}

pedestrian_det::pedestrian_det(int argc, char ** argv)
{
    printf("\nname =%s\n", argv[0]);
    initialized = -1;
    updated_count = 0; //initializing before the init method

    if(init(argc, argv)) //returns 1 if all variables are properly initialized
    {
        initialized = 1;
    }
}

pedestrian_det * pedestrian_det::get_detetctor(int argc, char ** argv)
{
    printf("\nname =%s\n", argv[0]);

    if(!detector_obj)
    {
        detector_obj = new pedestrian_det(argc, argv);
    }

    return detector_obj;
}

void pedestrian_det::onMouse(int event, int x, int y, int, void *)
{
    if(!frame1.data)
    {
        return;
    }

    Mat img_input = frame1;

    if(event == cv::EVENT_LBUTTONDOWN)
    {
        int x_final = 0;
        int y_final = 0;

        if(x - (width_roi / 2) <= 0)
        {
            x_final = 1;
        }
        else if(x + (width_roi / 2) >= img_input.cols)
        {
            x_final = (x - (width_roi / 2)) - (x + (width_roi / 2) - (img_input.cols - 1));
        }
        else
        {
            x_final = x - (width_roi / 2);
        }

        if(y - (height_roi / 2) <= 0)
        {
            y_final = 1;
        }
        else if(y + (height_roi / 2) >= img_input.rows)
        {
            y_final = (y - (height_roi / 2)) - (y + (height_roi / 2) - (img_input.rows - 1));
        }
        else
        {
            y_final = y - (height_roi / 2);
        }

        x_roi = x_final;
        y_roi = y_final;
    }
}


