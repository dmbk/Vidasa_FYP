#include "main_processor.hpp"

vector<Point> vehicle_det::ROI_Vertices;
Mat vehicle_det::frame0;
vehicle_det * vehicle_det::detector_obj = NULL;
int vehicle_det::min_x;
int vehicle_det::min_y;
int vehicle_det::max_x;
int vehicle_det::max_y;
int vehicle_det::MAX_VEH_DENSITY = 30;
double vehicle_det::veh_score_red_rate = 5.0;

void vehicle_det::define_polygon(Mat & img)
{
    cout << __PRETTY_FUNCTION__ << endl;
    frame0 = img;
    cv::namedWindow("select_roi", CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback("select_roi", onMouse, 0);
    printf("Click on %d points to define the polygon\n", POLYGON_CORNERS);

    while(ROI_Vertices.size() < 4) //press any other key to proceed to next frame
    {
        imshow("select_roi", frame0);
        cv::waitKey(30);
    }

    main_roi = Rect(min_x, min_y, max_x - min_x, max_y - min_y);
    waitKey(100);
    cvDestroyAllWindows();
}

void vehicle_det::process_equation(double input_val)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    std::unique_lock<std::mutex> lk(abstract_det::m);
    cout <<endl<< "===veh proc eq===" << endl;
    cout << "max veh density " << vehicle_det::MAX_VEH_DENSITY << endl;
    cout << "veh_score_red_rate " << vehicle_det::veh_score_red_rate << endl;
    cout << "veh- Total_Score " << abstract_det::Total_Score << endl;
    double wc = (2.0 / (1.0 + (input_val / vehicle_det::MAX_VEH_DENSITY))) * vehicle_det::veh_score_red_rate;
    cout << "veh- wc " << wc << endl << endl;
    abstract_det::Total_Score = abstract_det::Total_Score - floor(wc);
    lk.unlock();
}
int vehicle_det::do_iteration()
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    cv::Mat img_input, src;
    cap >> src;

    if(!src.data)
    {
        printf("Exiting\n");
        return -1;
    }

    Mat img_display = src.clone();
    draw_ROI_poly(img_display);
    src.copyTo(img_input, mask);
    img_input = Mat(img_input, main_roi);
    IplImage temp = img_input;
    IplImage * frame = &temp;
    //getting the polygon
    // bgs->process(...) internally process and show the foreground mask image
    cv::Mat img_mask;
    //bgs->process(img_input, img_mask);
    get_foreground(img_input, img_mask);
    blur(img_mask, img_mask, Size(4, 4));
    img_mask = img_mask > 10;
    /*morphologyEx(img_mask, img_mask, MORPH_CLOSE, Mat(25, 2, CV_8U));
    morphologyEx(img_mask, img_mask, MORPH_OPEN, Mat(10, 10, CV_8U));*/
    morphologyEx(img_mask, img_mask, MORPH_CLOSE, Mat(2, 2, CV_8U));
    //morphologyEx(img_mask, img_mask, MORPH_OPEN, Mat(10, 10, CV_8U));
    //morphologyEx(img_mask, img_mask, MORPH_GRADIENT , Mat(5,5, CV_8U));
    //bgs->operator()(img_input,img_mask,0.2);
    //erode(img_mask, img_mask, Mat());
    //dilate(img_mask, img_mask, Mat());
    //imshow("fore", img_mask);

    if(!img_mask.empty())
    {
        //vector<Rect> rois;// to be added all the ROIs
        IplImage copy = img_mask;
        IplImage * new_mask = &copy;
        IplImage * labelImg = cvCreateImage(cvGetSize(new_mask), IPL_DEPTH_LABEL, 1);
        CvBlobs blobs, filtered_blobs;
        unsigned int result = cvb::cvLabel(new_mask, labelImg, blobs);
        cvFilterByArea(blobs, 40, 2000);
        int count = 0;

        for(CvBlobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
        {
            count++;
            //  cout << "Blob #" << it->second->label << ": Area=" << it->second->area << ", Centroid=(" << it->second->centroid.x << ", " << it->second->centroid.y << ")" << endl;
            int x, y;
            x = (int)it->second->centroid.x;
            y = (int)it->second->centroid.y;
            //cv::Point2f p(x,y );
            // circle(img_input, p, (int)10, cv::Scalar(255, 0 , 0), 2, 8, 0);
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

            //printf("resized x_final=%d y_final=%d  cols=%d,  rows=%d \n", x_final,y_final,img_input.cols,img_input.rows);
            Rect roi(x_final, y_final, width_roi, height_roi);
            //rois.push_back(roi);//adding ROIs using rectangles
            //		Mat image = imread("");
            Mat image_roi = Mat(img_input, roi);
            int vehicle_ct = detect(image_roi); //getting the vehicle count per ROI

            if(vehicle_ct > 0)
            {
                filtered_blobs[it->first] = it->second;
                int matched = 0;
                int c1 = 255, c2 = 0;

                if(matched)
                {
                    c1 = 0;
                    c2 = 255;
                }
                else
                {
                    //print something to debug
                }//changing the colour of  the rectanged depending on matched or not matched

                rectangle(img_display,
                          Point(min_x + x - 5, min_y + y - 5),
                          Point(min_x + x + 5, min_y + y + 5),
                          CV_RGB(c1, c2, 0), 2, 8, 0);
                /*rectangle(img_input,
                          Point(x - 5, y - 5),
                          Point(x + 5, y + 5),
                          CV_RGB(c1, c2, 0), 2, 8, 0);*/
            }
        }

        //cvUpdateTracks(filtered_blobs, tracks, 5., 10);
        cvUpdateTracks(filtered_blobs, tracks, 10., 5);
        cvRenderBlobs(labelImg, filtered_blobs, frame, frame, CV_BLOB_RENDER_CENTROID | CV_BLOB_RENDER_BOUNDING_BOX);
        //cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX|CV_TRACK_RENDER_TO_LOG);
        cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID);
        printf("num of active tracks %d\n", tracks.size());
        process_equation(tracks.size());//number of people given as input
	if(abstract_det::Total_Score<0){
		destroyAllWindows();	
	}
    }

    if(!img_display.empty())
    {
        cv::imshow("vehicle_view", img_display);
    }

    waitKey(30);
    
    return 1;
}

void vehicle_det::start_session()
{
    cout << __PRETTY_FUNCTION__ << endl;

    while(do_iteration())
    {
        printf("Vehicle detection score:");
    }
}
void vehicle_det::draw_ROI_poly(Mat & img)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    int i = 0;

    for(i; i < ROI_Vertices.size(); ++i)
    {
        if(i + 1 == ROI_Vertices.size())
        {
            draw_line(img, ROI_Vertices[i], ROI_Vertices[0]);
            break;
        }

        draw_line(img, ROI_Vertices[i], ROI_Vertices[i + 1]);
    }
}
void vehicle_det::draw_line(Mat & img, Point & start, Point & end)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    int thickness = 2;
    int lineType = 8;
    line(img,
         start,
         end,
         Scalar(0, 255, 0),
         thickness,
         lineType);
}
Mat vehicle_det::get_mask(Mat & src)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    /* ROI by creating mask for the parallelogram */
    Mat mask = Mat(src.rows, src.cols, CV_8UC1);

    // Create black image with the same size as the original
    for(int i = 0; i < mask.cols; i++)
        for(int j = 0; j < mask.rows; j++)
        {
            mask.at<uchar>(Point(i, j)) = 0;
        }

    // Create Polygon from vertices
    vector<Point> ROI_Poly;
    approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);
    // Fill polygon white
    fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
    // Cut out ROI and store it in imageDest
    return mask;
}

void vehicle_det::get_foreground(Mat & background, Mat & foreground)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    mog(background, foreground, -1);
    threshold(foreground, foreground, 175, 255, THRESH_BINARY);
    //threshold(foreground, foreground, 150, 255, THRESH_TOZERO);
    medianBlur(foreground, foreground, 9);
    erode(foreground, foreground, Mat());
    dilate(foreground, foreground, Mat());
}



/*
shadow removal algorithm
===============================
1.Remove shadows by converting to HSV and setting V to a fixed value
2.Convert to grayscale and normalize
3.Apply gaussian blur and Canny edge detector
4.Dilate to close gaps
5.Flood fill the image from borders
6.Erode to account for previous dilation
7.Find largest contour
8.Mask original image
*/
void vehicle_det::remove_shadows()
{
    //cout<<__PRETTY_FUNCTION__<<endl;
}



int vehicle_det::detect(Mat & detectable)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
    vector<Rect> veh;
    /*    CvSeq * object = cvHaarDetectObjects(
                             img,
                             cascade,
                             storage,
                             1.007, //1.1,//1.5, //-------------------SCALE FACTOR
                             1, //2        //------------------MIN NEIGHBOURS
                             0, //CV_HAAR_DO_CANNY_PRUNING
                             cvSize(0, 0), //cvSize( 30,30), // ------MINSIZE
                             img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
                         );
    */
    cascade.detectMultiScale(detectable, veh, 1.01, 3, 0, Size(width_roi / 4, height_roi / 4), detectable.size());
    return veh.size();
}



void vehicle_det::onMouse(int event, int x, int y, int, void *)
{
    //cout<<__PRETTY_FUNCTION__<<endl;
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
    cout << __PRETTY_FUNCTION__ << endl;

    if(!detector_obj)
    {
        detector_obj = new vehicle_det(argc, argv);
    }

    return detector_obj;
}

vehicle_det::vehicle_det(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
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
    cout << __PRETTY_FUNCTION__ << endl;
    /* Background Subtraction Algorithm */
    //IBGS * bgs;
    //bgs = new PixelBasedAdaptiveSegmenter;
    //BackgroundSubtractorMOG2 *bgs=new BackgroundSubtractorMOG2(200, 25, false);
    /* Open video file */
    vehicle_det::MAX_VEH_DENSITY = 30;
    vehicle_det::veh_score_red_rate = 5.0;
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

    if(argc >= 4)
    {
        width_roi = atoi(argv[2]);
        height_roi = atoi(argv[3]);
    }

    if(argc == 5)
    {
        MAX_VEH_DENSITY = atoi(argv[4]);
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
int vehicle_det::continue_display(){
    Mat src;
    cap >> src;

    if(!src.data)
    {
	return -1;
        
    }
    imshow("vehicle_view",src);
	cv::waitKey(25);
	return 1;
}


