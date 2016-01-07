#include <iostream>
#include <cv.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <highgui.h>
#include <cvblob.h>
#include "pedestrian_det.hpp"
//#include "PixelBasedAdaptiveSegmenter.h"
#include <limits.h>

//#include "package_tracking/BlobTracking.h"
//#include "package_analysis/VehicleCouting.h"
using namespace cvb;
using namespace std;
using namespace cv;



#define POLYGON_CORNERS 4



class vehicle_det:public abstract_det
{

    private:
        static vector<Point> ROI_Vertices;
        static Mat frame0;//first frame to define the polygon
	static int MAX_VEH_DENSITY;
	static double veh_score_red_rate;
	
        static vehicle_det * detector_obj;
        static int min_x;
        static int min_y;
        static int max_x;
        static int max_y;

        int initialized;
       
        int width_roi;
        int height_roi;//detection window size for vehicles (roi size to consider for haar feature verification)
	int updated_count;

        CascadeClassifier cascade;//for new version
        BackgroundSubtractorMOG2 mog;
        VideoCapture cap;
        CvTracks tracks;//for blob tracking
        Mat mask;
        Rect main_roi;


        vehicle_det(int argc, char ** argv);
        int detect(Mat & detectable);
        void get_foreground(Mat & background, Mat & foreground);
        void remove_shadows();
        void define_polygon(Mat & img);
        Mat get_mask(Mat & src);
        int init(int argc, char ** argv); //only supposed to be initialized via the constructor
        void draw_ROI_poly(Mat & img);
        void draw_line(Mat & img, Point & start, Point & end);
	int process_equation(int input_val);

        static void onMouse(int event, int x, int y, int, void *);

	
    protected:
	int process_equation();
    public:
        static vehicle_det * get_detetctor(int argc, char ** argv);

        void start_session();
	int do_iteration();
	int get_updated_count()
        {
            int temp = updated_count;
            updated_count = 0;
            return temp;
        }
      


};

	

void vehicle_det::onMouse(int event, int x, int y, int, void *)
{
    if(!frame0.data)
    {
        return;
    }

    Mat img_input = frame0;

    if(event == cv::EVENT_LBUTTONDOWN)
    {
        ROI_Vertices.push_back(Point(x, y));
        circle(frame0, Point(x, y), 1, CV_RGB(0, 255, 255));

        if(min_x > x)
        {
            min_x = x;
        }

        if(min_y > y)
        {
            min_y = y;
        }

        if(max_x < x)
        {
            max_x = x;
        }

        if(max_y < y)
        {
            max_y = y;
        }
    }
}



vehicle_det * vehicle_det::get_detetctor(int argc, char ** argv)
{
    if(!detector_obj)
    {
        detector_obj = new vehicle_det(argc, argv);
    }

    return detector_obj;
}

vehicle_det::vehicle_det(int argc, char ** argv)
{
    initialized = -1;
    updated_count = 0; //initializing before the init method
    width_roi = 65;
    height_roi = 65;

    if(init(argc, argv)) //returns 1 if all variables are properly initialized
    {
        initialized = 1;
    }
}

int vehicle_det::init(int argc, char ** argv)
{
    /* Background Subtraction Algorithm */
    //IBGS * bgs;
    //bgs = new PixelBasedAdaptiveSegmenter;
    //BackgroundSubtractorMOG2 *bgs=new BackgroundSubtractorMOG2(200, 25, false);
    /* Open video file */
    cap.open(argv[1]);

    if(!cap.isOpened())
    {
        std::cerr << "Cannot open video!" << std::endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    if(!cascade.load("data/veh.xml"))
    {
        printf("xml file not loaded");
        return -1;
    }

    if(argc >=4)
    {
        width_roi = atoi(argv[2]);
        height_roi = atoi(argv[3]);
	
    }

    if(argc ==5){
	MAX_VEH_DENSITY=atoi(argv[4]);
    }else{

	return -1;
    }
    
    
    Mat first_mat;
    cap >> first_mat;
    min_x = first_mat.cols - 1;
    min_y = first_mat.rows - 1;
    max_x = 0;
    max_y = 0; //initialising the main_roi with default parameters
    printf("min_x %d    min_y %d\n", min_x, min_y);
    define_polygon(first_mat);
    mask = get_mask(first_mat);

    if(!mask.data)
    {
        printf("null\n");
    }
}


