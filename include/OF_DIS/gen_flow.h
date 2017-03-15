
// Class implements main flow computation loop over all scales

#ifndef GEN_FLOW_HEADER
#define GEN_FLOW_HEADER

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
    
#include "OF_DIS/oflow.h"

void gen_flow( cv::Mat img_ao_mat, cv::Mat img_bo_mat, cv::Mat *flowout, int sel_oppoint = 0);
void image_warp(cv::Mat& img, cv::Mat& flow, cv::Mat& ret, cv::Mat& mask);
#endif /* GEN_FLOW_HEADER */


