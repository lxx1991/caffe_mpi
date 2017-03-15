#include "OF_DIS/gen_flow.h"

#define MIN_TA(a, b) ((a) < (b) ? (a) : (b))
#define MAX_TA(a, b) ((a) > (b) ? (a) : (b))
#define MINMAX_TA(a,b) MIN_TA( MAX_TA(a,0) , b-1 )


void image_warp(cv::Mat& img, cv::Mat& flow, cv::Mat& ret, cv::Mat& mask)
{
  cv::Size szt = flow.size();
  int width = szt.width, height = szt.height;
  int nc = flow.channels();

  if (nc == 2)
  {
    if (img.channels() == 3)
      ret.create(height, width, CV_8UC3);
    else
      ret.create(height, width, CV_32FC1);
    mask = cv::Mat(height, width, CV_32FC1);

    int x, y, x1, x2, y1,y2;
    float xx, yy, dx, dy;
    
    for (int j=0; j<height; j++)
      for (int i=0; i<width; i++)
      {
          xx = i+flow.at<cv::Vec2f>(j,i)[0];
          yy = j+flow.at<cv::Vec2f>(j,i)[1];
          
          x = floor(xx);
          y = floor(yy);
          dx = xx-x;
          dy = yy-y;

          mask.at<float>(j,i) = (xx>=0 && xx<=width-1 && yy>=0 && yy<=height-1);

          x1 = MINMAX_TA(x,width);
          x2 = MINMAX_TA(x+1,width);
          y1 = MINMAX_TA(y,height);
          y2 = MINMAX_TA(y+1,height);

          if (img.channels() == 3)
            for (int k=0; k<3; k++)
            {

              ret.at<cv::Vec3b>(j,i)[k] = 
                  (uchar)((float)img.at<cv::Vec3b>(y1,x1)[k]*(1.0f-dx)*(1.0f-dy) +
                  (float)img.at<cv::Vec3b>(y1,x2)[k]*dx*(1.0f-dy) +
                  (float)img.at<cv::Vec3b>(y2,x1)[k]*(1.0f-dx)*dy +
                  (float)img.at<cv::Vec3b>(y2,x2)[k]*dx*dy);
            }
          else
              ret.at<float>(j,i) = 
                  ((float)img.at<float>(y1,x1)*(1.0f-dx)*(1.0f-dy) +
                  (float)img.at<float>(y1,x2)*dx*(1.0f-dy) +
                  (float)img.at<float>(y2,x1)*(1.0f-dx)*dy +
                  (float)img.at<float>(y2,x2)*dx*dy);
      }
  }
}


void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh)
{
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
        img_ao_fmat_pyr[i] = img_ao_fmat.clone();
      else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
	      
      img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);
	
      if ( getgrad ) 
      {
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
        img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
        img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
      }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
      img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

      if ( getgrad ) 
      {
        copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // Zero padding for gradients
        copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

        img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
        img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
      }
    }
}

int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
{
  return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
}

void gen_flow( cv::Mat img_ao_mat, cv::Mat img_bo_mat, cv::Mat *flowout, int sel_oppoint)
{
  int rpyrtype = CV_32FC3;


  cv::Mat img_tmp;

  cv::Mat img_ao_fmat, img_bo_fmat;
  cv::Size sz = img_ao_mat.size();
  
  int nochannels = 3; 
  int width_org = sz.width;
  int height_org = sz.height;

  
  // *** Parse rest of parameters, See oflow.h for definitions.
  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
  bool usefbcon, usetvref;
  //bool hasinfile; // initialization flow file
  //char *infile = nullptr;
  
  mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;    
  usefbcon = 0; patnorm = 1; costfct = 0; 
  tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
  tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
      
  int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
    
  switch (sel_oppoint)
  {
    case 1:
      patchsz = 8; poverl = 0.3; 
      lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
      lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16; 
      usetvref = 0; 
      break;
    case 3:
      patchsz = 12; poverl = 0.75; 
      lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
      lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16; 
      usetvref = 1; 
      break;
    case 4:
      patchsz = 12; poverl = 0.75; 
      lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
      lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128; 
      usetvref = 1; 
      break;        
    case 2:
    default:
      patchsz = 8; poverl = 0.4; 
      lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
      lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12; 
      usetvref = 1; 
      break;

  }

  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  //if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;          
  if (padh>0 || padw>0)
  {
    copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
    copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
  
  // Timing, image loading
  /*if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Image loading     ) (ms): %3g\n", tt);
    gettimeofday(&tv_start_all, NULL);
  }*/
  
  
  
  
  //  *** Generate scale pyramides
  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);
  
  const float* img_ao_pyr[lv_f+1];
  const float* img_bo_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  const float* img_bo_dx_pyr[lv_f+1];
  const float* img_bo_dy_pyr[lv_f+1];
  
  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_bo_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];
  
  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);

  // Timing, image gradients and pyramid
  /*if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Pyramide+Gradients) (ms): %3g\n", tt);
  }*/

  
//     // Read Initial Truth flow (if available)
//     float * initptr = nullptr;
//     cv::Mat flowinit;
//     if (hasinfile)
//     {
//       #if (SELECTMODE==1)
//       flowinit.create(height_org, width_org, CV_32FC2);
//       #else
//       flowinit.create(height_org, width_org, CV_32FC1);
//       #endif
//       
//       ReadFlowFile(flowinit, infile);
//         
//       // padding to ensure divisibility by 2
//       if (padh>0 || padw>0)
//         copyMakeBorder(flowinit,flowinit,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
//       
//       // resizing to coarsest scale - 1, since the array is upsampled at .5 in the code
//       float sc_fct = pow(2,-lv_f-1);
//       flowinit *= sc_fct;
//       cv::resize(flowinit, flowinit, cv::Size(), sc_fct, sc_fct , cv::INTER_AREA); 
//       
//       initptr = (float*)flowinit.data;
//     }

  
  
  
  //  *** Run main optical flow / depth algorithm
  float sc_fct = pow(2,lv_l);
  flowout->create(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2); // Optical Flow
      
  
  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
                    img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
                    patchsz,  // extra image padding to avoid border violation check
                    (float*)flowout->data,   // pointer to n-band output float array
                    nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
                    sz.width, sz.height, 
                    lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
                    usefbcon, costfct, nochannels, patnorm, 
                    usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                    0);    

  //if (verbosity > 1) gettimeofday(&tv_start_all, NULL);
      
  
  
  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    (*flowout) *= sc_fct;
    cv::resize(*flowout, *flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before saving to file
  *flowout = (*flowout)(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));


  //fid = fopen(outfile, "wb");
  //assert((int)fwrite(flowout.data, sizeof(float), height_org * width_org * 2, fid) == height_org * width_org * 2);
  //fclose(fid);
  //SaveFlowFile(flowout, outfile);
  //cv::Mat ret;
  //cv::Mat mask;
  //image_warp(img_origin, flowout, ret, mask);
  //cv::imwrite("warp_img.png", ret);
  //cv::imwrite("mask.png", mask);

  /*if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);
  }*/
}


    


