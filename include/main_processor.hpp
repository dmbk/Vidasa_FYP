#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include<stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "cv.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include "opencv2/core/core.hpp"
#include <cvblob.h>

//#include "PixelBasedAdaptiveSegmenter.h"
#include <limits.h>

using namespace cvb;
using namespace std;
using namespace cv;

class vehicle_det;
class pedestrian_det;
class abstract_det;
class main_processor;

class abstract_det
{
    protected:
        virtual void process_equation(double input_val) = 0;
    public:

        //for multi-threaded execution purposes
        static std::mutex m;
        static std::condition_variable cv;
        static long Total_Score;
        static void init_total_score();
        static int _session_started;

        static void set_session_started(int val);
        static int session_started();

	virtual int continue_display()=0;
        virtual void start_session() = 0;
        virtual int do_iteration() = 0;
        virtual int get_updated_count() = 0;

};


#define POLYGON_CORNERS 4



class vehicle_det: public abstract_det
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
        static void onMouse(int event, int x, int y, int, void *);
    protected:
        void process_equation(double input_val);





    public:
        static vehicle_det * get_detetctor(int argc, char ** argv);
	int continue_display();
        void start_session();
        int do_iteration();
        int get_updated_count()
        {
            int temp = updated_count;
            updated_count = 0;
            return temp;
        }



};

class pedestrian_det: public abstract_det
{

    private:
        static pedestrian_det * detector_obj;

        static const int width_roi = 400;
        static const int height_roi = 200; //assumes these are lesser than the image width and height
        static int x_roi, y_roi;
        static Mat frame1;
        static int MAX_PED_DENSITY;
        static double ped_score_red_rate;

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
        void process_equation(double input_val);
    public:
        static pedestrian_det * get_detetctor(int argc, char ** argv);
        int continue_display();
        void start_session();
        int do_iteration();
        int get_updated_count()
        {
            int temp = updated_count;
            updated_count = 0;
            return temp;
        }



};

class main_processor
{
    private:
        abstract_det * pedestrian_detector;
        abstract_det * vehicle_detector;
        static main_processor * proc;
        main_processor(int argc, char ** argv);
    public:
        ~main_processor();
        static main_processor * init_processor(int argc, char ** argv);
        void do_process();

};


