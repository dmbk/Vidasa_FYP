#include "main_processor.hpp"

int pedestrian_det::x_roi = -1, pedestrian_det::y_roi = -1;
pedestrian_det * pedestrian_det::detector_obj = NULL;
Mat pedestrian_det::frame1;
int pedestrian_det::MAX_PED_DENSITY = 20;
double pedestrian_det::ped_score_red_rate = 5.0;


int pedestrian_det::do_iteration()
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    Mat col_img_raw;
    Mat img, img_temp;
    cap >> img_temp;

    if(!img_temp.data)
    {
        return -1;
    }

    Mat img_display = img_temp.clone();
    draw_ROI_poly(img_display);
    img = Mat(img_temp, main_roi);
    //"/home/dulitha/opencv-3.0.0-rc1/data/haarcascades/haarcascade_lowerbody.xml"
    vector<Rect> pedestrians;
    //cascade.detectMultiScale(img, faces, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING, Size(30, 30), Size(200, 200));
    //cascade.detectMultiScale(img, pedestrians, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30), Size(200, 200));
    cascade.detectMultiScale(img, pedestrians, 1.1, 1, CV_HAAR_DO_CANNY_PRUNING, Size(30, 30), Size(150, 150));

    for(int i = 0; i < pedestrians.size(); i++)
    {
        Mat image_roi = Mat(img, pedestrians[i]);
        int votes = 1	;
        //votes = calc_votes(image_roi);

        if(votes >= 1)
        {
            //printf("rows=%d cols=%d",image_roi.rows,image_roi.cols);
            Point center(main_roi.x + pedestrians[i].x + pedestrians[i].width * 0.5, main_roi.y + pedestrians[i].y + pedestrians[i].height * 0.5);
            ellipse(img_display, center, Size(pedestrians[i].width * 0.5, pedestrians[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
            updated_count++;
        }
    }

    if(!img_display.empty())
    {
        imshow("pedestrian_view", img_display);
    }

    if(abstract_det::session_started() == 1)
    {
        process_equation(pedestrians.size());//number of pedestrians given as the input
	if(abstract_det::Total_Score<0){
		destroyAllWindows();	
	}
    }

    waitKey(30);
    
    return 1;
}

void pedestrian_det::process_equation(double input_val)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    std::unique_lock<std::mutex> lk(abstract_det::m);
    cout <<endl<< "===ped proc eq===" << endl;
    cout << "max ped density " << pedestrian_det::MAX_PED_DENSITY << endl;
    cout << "ped_score_red_rate " << pedestrian_det::ped_score_red_rate << endl;
    cout << "ped -Total_Score " << abstract_det::Total_Score << endl;
    double wc = (1.0 + (input_val / pedestrian_det::MAX_PED_DENSITY)) * pedestrian_det::ped_score_red_rate;
    cout << "ped- wc " << wc << endl;
    abstract_det::Total_Score = abstract_det::Total_Score - floor(wc);
    lk.unlock();
}

void pedestrian_det::start_session()
{
    cout << __PRETTY_FUNCTION__ << endl;

    while(do_iteration())
    {
        printf("Pedestrian detection score:");
    }
}

Rect pedestrian_det::get_roi(Mat & img)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    frame1 = img;
    cv::namedWindow("select_roi", CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback("select_roi", onMouse, 0);
    printf("Click on the center of the ROI\n");

    while(x_roi < 0 && y_roi < 0) //press any other key to proceed to next frame
    {
        imshow("select_roi", frame1);
        cv::waitKey(30);
    }

    cvDestroyAllWindows();
    Rect roi(x_roi, y_roi, width_roi, height_roi);
    return roi;
}
void pedestrian_det::draw_ROI_poly(Mat & img)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    rectangle(img,
              Point(main_roi.x, main_roi.y),
              Point(main_roi.x + main_roi.width, main_roi.y + main_roi.height),
              CV_RGB(0, 255, 0), 2, 8, 0);
}
int pedestrian_det::detect_hog_svm(Mat & img)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    vector<Rect> found, found_filtered;
    hog.detectMultiScale(img, found, -0.05, Size(4, 4), Size(32, 32), 1.1, 2);
    size_t i, j;

    for(i = 0; i < found.size(); i++)
    {
        Rect r = found[i];

        for(j = 0; j < found.size(); j++)
            if(j != i && (r & found[j]) == r)
            {
                break;
            }

        if(j == found.size())
        {
            found_filtered.push_back(r);
        }
    }

    return found_filtered.size();
}


int pedestrian_det::init(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
    pedestrian_det::x_roi = -1;
    pedestrian_det::y_roi = -1;
    pedestrian_det::MAX_PED_DENSITY = 25;
    cout<<"init pedestrian_det::MAX_PED_DENSITY"<<pedestrian_det::MAX_PED_DENSITY<<endl;
    pedestrian_det::ped_score_red_rate = 5.0;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    if(!cascade.load("data/ped.xml"))
    {
        printf("xml file not loaded");
        return -1;
    }

    cap = VideoCapture(argv[1]);

    if(!cap.isOpened())
    {
        printf("init returning ped\n");
        return -1;
    }

    if(argc == 3)
    {
        MAX_PED_DENSITY = atoi(argv[2]);
        cout << argv[2] << " arg2 " << endl;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    Mat image_first;
    cap >> image_first;
    main_roi = get_roi(image_first); //defined using mouse click
}

pedestrian_det::pedestrian_det(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
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
    cout << __PRETTY_FUNCTION__ << endl;
    printf("\nname =%s\n", argv[0]);

    if(!detector_obj)
    {
        detector_obj = new pedestrian_det(argc, argv);
    }

    return detector_obj;
}

void pedestrian_det::onMouse(int event, int x, int y, int, void *)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
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

int pedestrian_det::continue_display(){
    Mat src;
    cap >> src;
if(!src.data)
    {
	return -1;
        
    }
 
        imshow("pedestrian_view",src);
	cv::waitKey(25);
  return 1;
    
}

