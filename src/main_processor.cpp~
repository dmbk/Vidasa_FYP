#include "main_processor.hpp"
//#include "pedestrian_det.hpp"
#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>

using namespace std;
using namespace cv;

long abstract_det::Total_Score = 1000;
main_processor * main_processor::proc = NULL;
std::mutex abstract_det::m;
std::condition_variable abstract_det::cv;
int abstract_det::_session_started = 0;


void DrawProgressBar(int len);//utility function

void abstract_det::init_total_score()
{
    cout << __PRETTY_FUNCTION__ << endl;
    //read from the file and initialize
    abstract_det::Total_Score = 1000;
}
void abstract_det::set_session_started(int val)
{
    _session_started = val;
}
int abstract_det::session_started()
{
    return _session_started;
}

main_processor::main_processor(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
    string width_roi = "80";
    string height_roi = "80";
    char * argvs_veh[] = {argv[0], argv[1], &width_roi[0], &height_roi[0]};
    /*1-veh_vid_path
      2-width_roi
      3-height_roi
      4-MAX_VEH_DENSITY
    */
    vehicle_detector = vehicle_det::get_detetctor(4, argvs_veh);
    char * argvs_ped[] = {argv[0], argv[2]};
    /*1-ped_vid_path
      2-MAX_PED_DENSITY
    */
    pedestrian_detector = pedestrian_det::get_detetctor(2, argvs_ped);
}
main_processor::~main_processor()
{
    delete pedestrian_detector;
    delete vehicle_detector;
}
main_processor * main_processor::init_processor(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
    abstract_det::init_total_score();
    abstract_det::_session_started = 0;
    main_processor::proc = new main_processor(argc, argv);
    return proc;
}

void main_processor::do_process()
{
    cout << __PRETTY_FUNCTION__ << endl;

    while(true)
    {
        while(pedestrian_detector->get_updated_count() == 0)
        {
           
            if(pedestrian_detector->do_iteration()<0){
		return;
	    }
        }

        abstract_det::set_session_started(1);//start of session

        while(abstract_det::Total_Score > 0)
        {
           
            if(pedestrian_detector->do_iteration()<0||vehicle_detector->do_iteration()<0){
		return;
	    }
	    
            
        }

        abstract_det::set_session_started(0);//end of session
         
	
	int lim_ped=5;
	cout<<"Red Light: Allow pedestrians to cross: "<<lim_ped<<" seconds allowed"<<endl;
	int i=lim_ped;
        int frame_factor=40;
	for(i;i>0;i--){
		
		
		//std::this_thread::sleep_for (std::chrono::seconds(1));
		DrawProgressBar(i);
		int y=0;
		for(y;y<frame_factor;y++){
			if(pedestrian_detector->continue_display()<0||vehicle_detector->continue_display()<0){
				return;
	    		}
			
            		
		}
	}
	cout<<endl;
        
	
        
	
	int lim_veh=5;
	cout<<endl<<"Green Light: Allow vehicles to go: "<<lim_veh<<"seconds allowed"<<endl;
	int j=lim_veh;
	for(j;j>0;j--){
		
		
		//std::this_thread::sleep_for (std::chrono::seconds(1));
		DrawProgressBar(j);
		int y=0;
		for(y;y<frame_factor;y++){
			if(pedestrian_detector->continue_display()<0||vehicle_detector->continue_display()<0){
				return;
	    		}
		}
	}
	cout<<endl;
        abstract_det::init_total_score();//re-initializing the total score 
	
        /*if(waitKey(500000) == 27)
        {
            break;
        }*/
    }
}
void DrawProgressBar(int len) {
  cout << "\x1B[2K"; // Erase the entire current line.
  cout << "\x1B[0E"; // Move to the beginning of the current line.
  string progress;
  for (int i = 0; i < len; ++i) {
    progress += "=";	
    
  }
  cout << "Remaining" << progress << len;
  flush(cout); // Required.
}

int main(int argc, char ** argv)
{
    cout << __PRETTY_FUNCTION__ << endl;
    main_processor * processor = main_processor::init_processor(argc, argv);
    processor->do_process();
    delete processor;
}
