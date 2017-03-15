#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#ifdef DTYPE_FP16
#include "caffe/half/half.hpp"
#define Dtype Half
#else
#define Dtype float
#endif

#ifdef USE_MOG
#include "caffe/util/MogControl.h"
#endif

#include "OF_DIS/gen_flow.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using namespace cv;


const float flow_th = 1.0f;
const float rgb_th = 0.002f;
const int grid_size = 15;
const float momentum_value = 0.0f;
const int duration = 1;
const int choose_num = 3;

// DEFINE_int32(gpu, 0,
//     "Run in GPU mode on given device ID.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "The pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(shrink, 8,
    "Optional; Supported interp shrink ratio.");
DEFINE_int32(show_channel, 1,
    "Optional; Mask input with channel id #showchannel.");
DEFINE_string(output_blob, "fc9_zoom",
    "Optional; The blob name for the output probability in network proto.");
DEFINE_string(video_input, "/DATA/video/clean/",
    "Optional; Input video path, or from camera.");
DEFINE_string(video_out, "",
    "output video file (postfix .avi is added).");
DEFINE_int32(fps, 24,
  "fps of output video");

class HumanPredictor {
public:
  HumanPredictor() {};
  ~HumanPredictor() {};
  bool Init(const std::string &net_file, const std::string &model_file, const bool is_GPU,
   const float* mean_ptr,  const std::string &output_blob, int shrink);
  bool PredictProb(const cv::Mat& input_frame, std::vector<cv::Mat>& human_prob_mat) const;
  bool PredictProb(const cv::Mat& input_frame, std::vector<cv::Mat>& human_prob_mat, std::vector<cv::Mat>& human_prob_mat_f, cv::Mat& human_mask_mat) const;
  vector<int> PadParam(int video_cols, int video_rows);
  Mat show_mask(const cv::Mat&  img_ori, const cv::Mat& prob_mat, int show_channel);
  inline int get_height_in() { return height_network_input; }
  inline int get_width_in() { return width_network_input; }
  inline int get_resize_width() { return resize_width_video; }
  inline int get_resize_height() { return resize_height_video; }
  inline int get_channel_network_prob() { return channel_network_prob; }

private:
  caffe::Net<float> *model;
  int height_network_input = 0;
  int width_network_input = 0;
  int height_network_prob = 0;
  int width_network_prob = 0;
  int channel_network_prob = 0;
  int resize_height_video = 0;
  int resize_width_video = 0;
  int m_shrink = 0;
  float mean[3]; // bgr mean
  string output_blob_name;
};

bool HumanPredictor::Init(const std::string &net_file, const std::string &model_file,
  const bool is_GPU, const float* mean_ptr, const std::string &output_blob, int shrink) {
  if (is_GPU >= 0) {
    LOG(INFO) << "Use GPU with device ID " << is_GPU;
    Caffe::SetDevice(is_GPU);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  model = new caffe::Net<float>(net_file, caffe::TEST);
  if (model == NULL) { return false; }
  model->CopyTrainedLayersFrom(model_file);

  output_blob_name = output_blob;
  boost::shared_ptr<caffe::Blob<float> > data_blob = model->blob_by_name("data");
  boost::shared_ptr<caffe::Blob<float> > prob_blob = model->blob_by_name(output_blob_name);

  m_shrink = shrink;

  CHECK_EQ(data_blob->num(), 1);
  CHECK_EQ(data_blob->channels(), 3);
  height_network_input = data_blob->height();
  width_network_input = data_blob->width();
  CHECK_GT(height_network_input, 0);
  CHECK_GT(width_network_input, 0);

  CHECK_EQ(prob_blob->num(), 1);
  CHECK_EQ(prob_blob->channels(), 2);
  height_network_prob = prob_blob->height();
  width_network_prob = prob_blob->width();
  channel_network_prob = prob_blob->channels();
  CHECK_GT(height_network_prob, 0);
  CHECK_GT(width_network_prob, 0);

  for (size_t c = 0; c < 3; c++) {
    mean[c] = mean_ptr[c];
  }

  LOG(INFO) << "predictor init finished";
  return true;
}

vector<int> HumanPredictor::PadParam(int video_cols, int video_rows){ 
  float scale = std::min((float)width_network_input / video_cols, (float)height_network_input / video_rows);

  resize_width_video = video_cols * scale;
  resize_height_video = video_rows * scale;

  vector<int> pad_vector(4, 0);
  pad_vector[0] = (height_network_input - resize_height_video) / 2;
  pad_vector[1] = height_network_input - resize_height_video - pad_vector[0];
  pad_vector[2] = (width_network_input - resize_width_video) / 2;
  pad_vector[3] = width_network_input - resize_width_video - pad_vector[2];

  LOG(INFO) << "Video Input: " << video_rows << " " << video_cols;
  LOG(INFO) << "Resized Video Input: " << resize_height_video << " " << resize_width_video;
  LOG(INFO) << "Pad Video Input: " << height_network_input << " " << width_network_input;
  LOG(INFO) << "Pad top/bottom/left/right: " << pad_vector[0] << " " << pad_vector[1] << " " << pad_vector[2] << " " << pad_vector[3];

  return pad_vector;
}

bool HumanPredictor::PredictProb(const cv::Mat& input_frame, std::vector<cv::Mat>& human_prob_mat) const
{
  boost::shared_ptr<caffe::Blob<float> > data_blob = model->blob_by_name("data");
  boost::shared_ptr<caffe::Blob<float> > prob_blob = model->blob_by_name(output_blob_name);

  CHECK_EQ(input_frame.rows, height_network_input);
  CHECK_EQ(input_frame.cols, width_network_input);
  CHECK_EQ(input_frame.channels(), 3);

  int output_channels = prob_blob->channels();
  human_prob_mat.resize(output_channels);
  for (int i = 0; i < output_channels; i++) {
    human_prob_mat[i] = Mat(height_network_prob, width_network_prob, CV_32FC1);
  }

  Mat bgrFrame[3];
  split(input_frame, bgrFrame);
  for (size_t c = 0; c < 3; c++) {
    bgrFrame[c].convertTo(bgrFrame[c], CV_32F);
    bgrFrame[c] = bgrFrame[c] - mean[c];
  }

  float* data_blob_data = data_blob->mutable_cpu_data();
  for (size_t c = 0; c < 3; c++) {
    memcpy(data_blob_data + c * height_network_input * width_network_input,
        bgrFrame[c].data, sizeof(float) * height_network_input * width_network_input);
  }

  Timer forward_timer;
  forward_timer.Start();
  model->ForwardPrefilled();
  forward_timer.Stop();
  LOG(INFO) << "Forward Time: " << forward_timer.MilliSeconds() << " ms.";

  const float* prob_blob_data = prob_blob->cpu_data();

  for (int c = 0; c < output_channels; c++) {
    memcpy(human_prob_mat[c].data, prob_blob_data + 
      c * height_network_prob * width_network_prob, 
      sizeof(float) * height_network_prob * width_network_prob);
  }
  return true;
}

bool HumanPredictor::PredictProb(const cv::Mat& input_frame, std::vector<cv::Mat>& human_prob_mat, std::vector<cv::Mat>& human_prob_mat_f, cv::Mat& human_mask_mat) const
{
  boost::shared_ptr<caffe::Blob<float> > data_blob = model->blob_by_name("data");
  boost::shared_ptr<caffe::Blob<float> > prob_blob = model->blob_by_name(output_blob_name);

  CHECK_EQ(input_frame.rows, height_network_input);
  CHECK_EQ(input_frame.cols, width_network_input);
  CHECK_EQ(input_frame.channels(), 3);

  int output_channels = prob_blob->channels();
  human_prob_mat.resize(output_channels);
  for (int i = 0; i < output_channels; i++) {
    human_prob_mat[i] = Mat(height_network_prob, width_network_prob, CV_32FC1);
  }

  Mat bgrFrame[3];
  split(input_frame, bgrFrame);
  for (size_t c = 0; c < 3; c++) {
    bgrFrame[c].convertTo(bgrFrame[c], CV_32F);
    bgrFrame[c] = bgrFrame[c] - mean[c];
  }

  float* data_blob_data = data_blob->mutable_cpu_data();
  for (size_t c = 0; c < 3; c++) {
    memcpy(data_blob_data + c * height_network_input * width_network_input,
        bgrFrame[c].data, sizeof(float) * height_network_input * width_network_input);
  }

  Timer forward_timer;
  forward_timer.Start();
  model->ForwardPrefilled();
  forward_timer.Stop();
  LOG(INFO) << "Forward Time: " << forward_timer.MilliSeconds() << " ms.";

  const float* prob_blob_data = prob_blob->cpu_data();

  for (int c = 0; c < output_channels; c++) {
    memcpy(human_prob_mat[c].data, prob_blob_data + 
      c * height_network_prob * width_network_prob, 
      sizeof(float) * height_network_prob * width_network_prob);
  }

  flip(human_prob_mat[0], human_prob_mat[0],1);
  flip(human_prob_mat[1], human_prob_mat[1],1); 

  human_mask_mat = Mat(height_network_prob, width_network_prob, CV_32FC1, Scalar(0));
  for (int i = 0; i < height_network_prob * width_network_prob; i++){
    int index = 0; int prob = human_prob_mat[0].at<float>(i) + human_prob_mat_f[0].at<float>(i);
    for (int c = 1; c < output_channels; c++){
      if (human_prob_mat[c].at<float>(i) + human_prob_mat_f[c].at<float>(i) > prob){
        prob = human_prob_mat[c].at<float>(i) + human_prob_mat_f[c].at<float>(i) ;
        index = c;
      }
    }
    human_mask_mat.at<float>(i) = index;
  }
  return true;
}

Mat HumanPredictor::show_mask(const cv::Mat& img_ori, const cv::Mat& prob_mat, int show_channel) {
  Mat prob_img, prob_mat1;
  resize(prob_mat, prob_mat1, Size(img_ori.cols, img_ori.rows));
  CHECK_EQ(img_ori.rows, prob_mat1.rows);
  CHECK_EQ(img_ori.cols, prob_mat1.cols);

  CHECK_EQ(img_ori.channels(), 3);
  CHECK_EQ(prob_mat.channels(), 1);

  vector<Mat> split_channels;
  split(img_ori, split_channels);
  show_channel =  show_channel;
  for (int i = 0; i < img_ori.rows * img_ori.cols; i++){
    split_channels[1].at<uchar>(i) = prob_mat1.at<float>(i) == show_channel ?
        split_channels[1].at<uchar>(i) : 0;
  }
  merge(split_channels, prob_img);

  return prob_img;
}

bool mk_file_dir(string filename) {
  int path_len = 0;
  struct stat statbuf;
  int dir_err = 0;
  string path;
  for (string::iterator it=filename.begin()+1; it!=filename.end(); ++it) {
    path_len++;
    if (*it=='/') {
      path = filename.substr(0, path_len);
      char *cpath = new char [path.length()+1];
      strcpy(cpath, path.c_str());
      if (stat(cpath, &statbuf) != -1) {
        if (!S_ISDIR(statbuf.st_mode)) {}
      }
      else {
        dir_err = mkdir(cpath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (dir_err < 0) {
          return false;
        }
      }
      delete[] cpath;
    }
  }
  return true;
}

int median(std::vector<int> &input)
{
  std::vector<int> temp(input);
  sort(temp.begin(), temp.end());
  return temp[int(temp.size() / 2)];
}

int main(int argc, char** argv) {
  // Initialization
  


  caffe::GlobalInit(&argc, &argv);

  // CNN Initilization
  const float mean[3] = {103.939, 116.779, 123.68};
  string proto_model = FLAGS_model;
  string weights = FLAGS_weights;
  CHECK(!proto_model.empty()) << "Model prototxt needed.";
  CHECK(!weights.empty()) << "Trained model file needed.";
  int video_cols = 576;
  int video_rows = 1024;

  HumanPredictor predictor;
  CHECK(predictor.Init(proto_model, weights, 0, mean, FLAGS_output_blob, FLAGS_shrink));


  char buff[200];
  int tot_segment = 0;
  for (int i=1; i<5997; i++)
  {
    sprintf(buff, "%s%d.mp4", FLAGS_video_input.c_str(), i);
    LOG(ERROR) << buff;
    VideoCapture capture;
    capture.open(std::string(buff));  // Open from file
    if (!capture.isOpened()) {
      fprintf(stderr, "can not open camera or file not exist!\n");
      continue;
    }

    Mat read_frame, temp_frame, prev_frame;
    vector<int> deltas(FLAGS_fps *duration, 0), pad = predictor.PadParam(video_cols, video_rows);

    int cnt = 0, left = 0, mean_num;
    float global_mean = 0;
    std::map<int, int> hash;
    while (capture.read(read_frame)) {

      mean_num = cv::mean(read_frame)[0];
      global_mean = (mean_num  + global_mean * cnt) / (cnt + 1);
      int delta = 0;
      if (cnt >= (capture.get(CV_CAP_PROP_FRAME_COUNT) - 10 * FLAGS_fps))
        break;
      if (cnt >= 10 * FLAGS_fps)
      {
        vector<Mat> human_prob_mat, human_prob_mat_f;
        Mat human_mask_mat;

        cv::imshow("video", read_frame);
        resize(read_frame, read_frame, Size(predictor.get_resize_width(), predictor.get_resize_height()));
        

        copyMakeBorder(read_frame, read_frame, pad[0], pad[1], pad[2], pad[3], 
         BORDER_CONSTANT, Scalar(mean[0], mean[1], mean[2]) );

        predictor.PredictProb(read_frame, human_prob_mat_f);
        flip(read_frame, read_frame, 1);
        predictor.PredictProb(read_frame, human_prob_mat, human_prob_mat_f, human_mask_mat);

        if (cnt > 10 * FLAGS_fps)
        {
          delta = (int)cv::sum(cv::abs(human_mask_mat - prev_frame))[0];
        }

        if (cv::sum(human_mask_mat)[0]  < human_mask_mat.total() * 0.1 || abs(mean_num - global_mean) > global_mean / 3)
        {
          left = duration * FLAGS_fps + 10;
        }
        human_mask_mat.copyTo(prev_frame);
       
        //LOG(ERROR) << global_mean << ' ' << mean_num << ' ' << left;
        //waitKey(10);
      }
      //total_delta += delta - deltas[cnt % (duration * FLAGS_fps)];
      deltas[cnt % (duration * FLAGS_fps)] = delta;
      //LOG(ERROR) << cnt << ' ' << left << ' ' << capture.get(CV_CAP_PROP_POS_FRAMES) << ' ' << delta;
      
      if (left == 0)
        hash[median(deltas)] = cnt;
      else
        left--;
      cnt++;
    }

    vector<int> choose;
    std::map<int, int>::iterator it = hash.end();
    while(!hash.empty())
    {
      it--;
      bool flag = true;
      for (int j=0; j<choose.size(); j++)
        if (abs(it->second - choose[j]) < FLAGS_fps * duration * 2)
          flag = false;

      if (flag)
        choose.push_back(it->second);
      
      if (it == hash.begin() || choose.size() == choose_num)
        break;
    }

    if (choose.size() > 0)
    {
      sort(choose.begin(), choose.end());
      capture.release();
      capture.open(std::string(buff));  // Open from file
      if (!capture.isOpened()) {
        fprintf(stderr, "can not open camera or file not exist!\n");
        continue;
      }
      int j = 0, key = 0;
      while (capture.read(read_frame))
      {
        if (j == choose[key] - FLAGS_fps * duration)
        {
          sprintf(buff, "/DATA/video/segment/%d/%d_%d.avi", (int)(tot_segment / 100), i, key);
          CHECK(mk_file_dir(buff)) << "mkdir error: " << buff;
          VideoWriter vw;
          vw.open(buff, CV_FOURCC('M', 'J', 'P', 'G'), FLAGS_fps,
                cv::Size(video_cols, video_rows), true);
          while(true)
          {
            cv::imshow("video", read_frame);
            vw << read_frame;
            waitKey(1);
            if (j==choose[key])
              break;
            capture.read(read_frame);
            j++;
          }
          vw.release();
          tot_segment++;
          if (++key == choose.size())
            break;
        }
        j++;
      }
    }
      

/*
    while (capture.read(read_frame)) {
      int delta = 0;
      //if (cnt > 30 * FLAGS_fps && cnt < (capture.get(CV_CAP_PROP_FRAME_COUNT) - 30 * FLAGS_fps))
      {

        vector<Mat> human_prob_mat, human_prob_mat_f;
        Mat human_mask_mat;

        read_frame.copyTo(temp_frame);
        resize(temp_frame, temp_frame, Size(predictor.get_resize_width(), predictor.get_resize_height()));
        copyMakeBorder(temp_frame, temp_frame, pad[0], pad[1], pad[2], pad[3], 
         BORDER_CONSTANT, Scalar(mean[0], mean[1], mean[2]) );

        predictor.PredictProb(temp_frame, human_prob_mat_f);
        flip(temp_frame, temp_frame, 1);
        predictor.PredictProb(temp_frame, human_prob_mat, human_prob_mat_f, human_mask_mat);

        if (cnt!=0)
          delta = (int)cv::sum(cv::abs(human_mask_mat - prev_frame))[0];
        human_mask_mat.copyTo(prev_frame);
        //LOG(ERROR) << delta << ' ' <<total_delta;
        
        Mat masked_img = predictor.show_mask(read_frame, human_mask_mat, FLAGS_show_channel);
        Mat newImage;

        cv::cvtColor(human_mask_mat, human_mask_mat, cv::COLOR_GRAY2BGR);
        human_mask_mat.convertTo(human_mask_mat, CV_8UC3);

        resize(human_mask_mat, human_mask_mat, Size(video_cols, video_rows));
    

        cv::hconcat(masked_img, human_mask_mat, newImage);
        cv::imshow("Masked Input and Mask", newImage);

        Mat prob_concat = human_prob_mat[0];
        for (int i = 1; i < predictor.get_channel_network_prob(); i++){
          cv::hconcat(prob_concat, human_prob_mat[1], prob_concat);
        }
        cv::imshow("Channel Probilities", prob_concat);


        waitKey();
      }
      //total_delta += delta - deltas[cnt % (duration * FLAGS_fps)];
      deltas[cnt % (duration * FLAGS_fps)] = delta;
     // hash[total_delta] = cnt++;
    }*/
    capture.release();

  }

/*
  
    bool is_video_out = false;
    VideoWriter vw;
    string video_name;

  caffe::GlobalInit(&argc, &argv);

  // CNN Initilization
  const float mean[3] = {103.939, 116.779, 123.68};
  int video_cols = 640;
  int video_rows = 480;

  std::swap(video_cols, video_rows);

  HumanPredictor predictor;
  CHECK(predictor.Init(proto_model, weights, 0, mean, FLAGS_output_blob, FLAGS_shrink));
  vector<int> pad = predictor.PadParam(video_cols, video_rows);

  // Video initialization
  VideoCapture capture;
  if (FLAGS_video_input.empty()){
    capture.open(0);  // open the camera
  } else {
    capture.open(FLAGS_video_input);  // Open from file

    if (!FLAGS_video_out.empty()) {
      is_video_out = true;
      CHECK_GT(FLAGS_fps, 0) << "fps of output video should be greater than 0.";
      video_name = FLAGS_video_out+".avi";
      CHECK(mk_file_dir(video_name)) << "mkdir error: " << video_name;
      vw.open(video_name.c_str(), CV_FOURCC('M', 'J', 'P', 'G'), FLAGS_fps,
          cv::Size(video_cols, video_rows), true);
      CHECK(vw.isOpened()) << "Video Open Error : " << video_name;
      LOG(INFO) << "Video Write to : " << video_name;
    }
  }
  if (!capture.isOpened()) {
    fprintf(stderr, "can not open camera or file not exist!\n");
    return -1;
  }

  capture.set(CV_CAP_PROP_FRAME_WIDTH, video_cols);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, video_rows);

  Mat read_frame, temp_frame, prev_frame, flow(predictor.get_height_in(), predictor.get_width_in(), CV_32FC2), delta_flow(predictor.get_height_in(), predictor.get_width_in(), CV_32FC1), delta_rgb(predictor.get_height_in(), predictor.get_width_in(), CV_32FC1);


  Mat momentum(predictor.get_height_in(), predictor.get_width_in(), CV_32FC1, Scalar(0.00)), history_rgb, history_flow; int flag = 0;
  vector<Mat> history_human_prob_mat;
  while (capture.read(read_frame)) {
    // Prepare Input
    resize(read_frame, read_frame, Size(video_cols, video_rows));
    read_frame.copyTo(temp_frame);
    resize(temp_frame, temp_frame, Size(predictor.get_resize_width(), predictor.get_resize_height()));
    copyMakeBorder(temp_frame, temp_frame, pad[0], pad[1], pad[2], pad[3], 
       BORDER_CONSTANT, Scalar(mean[0], mean[1], mean[2]) );

    // Predict by CNN
    vector<Mat> human_prob_mat;
    Mat human_mask_mat, orig_human_mask_mat;

    if (!flag)
    {
      temp_frame.copyTo(prev_frame);
      predictor.PredictProb(temp_frame, human_prob_mat, human_mask_mat);
      history_human_prob_mat.resize(human_prob_mat.size());
      human_prob_mat[0].copyTo(history_human_prob_mat[0]);
      human_prob_mat[1].copyTo(history_human_prob_mat[1]);
    }
    else
    {

      Timer flow_timer;
      flow_timer.Start();
      gen_flow( temp_frame, prev_frame, &flow, 2);
      flow_timer.Stop();
      LOG(INFO) << "Flow Time: " << flow_timer.MilliSeconds() << " ms.";
      
      for (int i=0; i <predictor.get_height_in(); i++)
        for (int j=0; j<predictor.get_width_in(); j++)
        {
          delta_flow.at<float>(i,j) = 0;
          for (int k=0; k<2; k++)
            delta_flow.at<float>(i,j) += flow.at<Vec2f>(i,j)[k] * flow.at<Vec2f>(i,j)[k];

          delta_rgb.at<float>(i,j) = 0;
          for (int k=0; k<3; k++)
            delta_rgb.at<float>(i,j) += (float)(temp_frame.at<Vec3b>(i,j)[k] - prev_frame.at<Vec3b>(i,j)[k])*(temp_frame.at<Vec3b>(i,j)[k] - prev_frame.at<Vec3b>(i,j)[k]) / (255.0 * 255.0);
        }
      
      if (flag == 1)
      {
        delta_rgb.copyTo(history_rgb);
        delta_flow.copyTo(history_flow);
      }
      cv::swap(delta_rgb, history_rgb);
      cv::swap(delta_flow, history_flow);
      
      delta_rgb = (delta_rgb + history_rgb) / 2.0f;
      delta_flow = (delta_flow + history_flow) / 2.0f;
      

      cv::medianBlur(delta_rgb, delta_rgb, 5);
      cv::medianBlur(delta_flow, delta_flow, 5);
      
      for (int i=0; i <predictor.get_height_in(); i++)
        for (int j=0; j<predictor.get_width_in(); j++)
        {
          delta_rgb.at<float>(i,j) = delta_rgb.at<float>(i,j) > rgb_th;
          delta_flow.at<float>(i,j) = delta_flow.at<float>(i,j) > flow_th;
        }
      
      Mat newImage, newImage1, newImage2;
      cv::blur(delta_flow, newImage1, Size(grid_size, grid_size));
      cv::blur(delta_rgb, newImage2, Size(grid_size, grid_size));
      temp_frame.copyTo(prev_frame);
      predictor.PredictProb(temp_frame, human_prob_mat, human_mask_mat, history_human_prob_mat, newImage1, newImage2, momentum, orig_human_mask_mat);
*/
    /*  for (int i=0; i <predictor.get_height_in(); i++)
        for (int j=0; j<predictor.get_width_in(); j++)
        {

          newImage1.at<float>(i,j) = delta_flow.at<float>(i,j);
          newImage2.at<float>(i,j) = momentum.at<float>(i,j);
        }
*/
      /*cv::hconcat(delta_flow, delta_rgb, newImage);
      cv::hconcat(newImage, momentum, newImage);
      cv::imshow("diff", newImage);
    }

    // Visualization and Output
    Mat masked_img = predictor.show_mask(read_frame, human_mask_mat, FLAGS_show_channel);
    Mat newImage;

    if (flag)
    {
      cv::hconcat(human_mask_mat, orig_human_mask_mat, human_mask_mat);
    }

    cv::imshow("Masked Input and Mask", human_mask_mat);
    cv::cvtColor(human_mask_mat, human_mask_mat, cv::COLOR_GRAY2BGR);
    human_mask_mat.convertTo(human_mask_mat, CV_8UC3);
    
    if (flag)
      resize(human_mask_mat, human_mask_mat, Size(video_cols*2, video_rows));
    else
      resize(human_mask_mat, human_mask_mat, Size(video_cols, video_rows));
    
    LOG(INFO) << "here";
    LOG(INFO) << read_frame.rows << ' ' << read_frame.cols;

    cv::hconcat(masked_img, human_mask_mat, newImage);
    cv::imshow("Masked Input and Mask", newImage);

    LOG(INFO) << "here";
    Mat prob_concat = human_prob_mat[0];
    for (int i = 1; i < predictor.get_channel_network_prob(); i++){
      cv::hconcat(prob_concat, human_prob_mat[1], prob_concat);
    }
    cv::imshow("Channel Probilities", prob_concat);

    if (is_video_out) {
      Mat tovideo;
      resize(masked_img, tovideo, Size(video_cols, video_rows));
      vw << tovideo;
    }

    if (waitKey(1) != -1)
      break;
    flag++;
  }*/
  LOG(INFO) << "Finished!";
  return 0;
}
