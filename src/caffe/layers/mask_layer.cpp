#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  threshold_ = this->layer_param_.mask_param().threshold();
  prev_mask_ = this->layer_param_.mask_param().prev_mask();
  ignore_label_ = this->layer_param_.mask_param().ignore_label();
  if (prev_mask_)
    CHECK_GE(bottom.size(), 2);
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num(), 1);

  for (int i=1; i<bottom.size(); i++)
  {
    //prev mask or label
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[0]->height(), bottom[i]->height());
    CHECK_EQ(bottom[0]->width(), bottom[i]->width());
    CHECK_EQ(bottom[i]->channels(), 1);
  }
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  top[1]->Reshape(bottom[0]->num(), 1, 2, bottom[0]->height() * bottom[0]->width() + 1);
  if (top.size() > 2)
    top[2]->Reshape(bottom[0]->num(), 1, 1, 1);
}


template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int cnt = 0;
  int spatial_dim = top[0]->count();
  int channels = bottom[0]->channels();
  
  const Dtype *prob = bottom[0]->cpu_data();
  const Dtype *prev_mask = NULL;
  const Dtype *label = NULL;

  if (prev_mask_)
  {
    prev_mask = bottom[1]->cpu_data();
    if (bottom.size() > 2)
      label = bottom[2]->cpu_data();
  }
  else
    if (bottom.size() > 1)
      label = bottom[1]->cpu_data();

  Dtype *mask = top[0]->mutable_cpu_data();
  Dtype *idx = top[1]->mutable_cpu_data();


  if (prev_mask != NULL && label != NULL)
  {
    for (int i = 0; i < bottom[0]->height(); i++)
      for (int j = 0; j < bottom[0]->width(); j++)
      {
        const int index = i * bottom[0]->width() + j;
        mask[index] = Dtype(-1);
        if (static_cast<int>(prev_mask[index]) == -1)
          continue;

        const int label_value = static_cast<int>(label[index]);
        if (label_value != ignore_label_)
        {
          //not ignore label
          if (prob[label_value * spatial_dim + index] > threshold_)
            continue;
        }
        else
        {
          //ignore label
          bool flag = false;
          for (int k=0; k < channels; k++)
            if (prob[k * spatial_dim + index] > threshold_)
            {
              flag = true;
              break;
            }
          if (flag) continue;
        }
        //keep pixel (i, j)
        idx[1 + cnt] = Dtype(i);
        idx[spatial_dim + 2 + cnt] = Dtype(j);
        mask[index] = Dtype(cnt++);
      }
  }
  else if (prev_mask != NULL && label == NULL)
  {
    for (int i = 0; i < bottom[0]->height(); i++)
        for (int j = 0; j < bottom[0]->width(); j++)
        {
          const int index = i * bottom[0]->width() + j;
          mask[index] = Dtype(-1);
          //prev mask
          if (static_cast<int>(prev_mask[index]) == -1)
            continue;

          //get max probability
          bool flag = false;
          for (int k=0; k < channels; k++)
            if (prob[k * spatial_dim + index] > threshold_)
            {
              flag = true;
              break;
            }
          if (flag) continue;
          //keep pixel (i, j)
          idx[1 + cnt] = Dtype(i);
          idx[spatial_dim + 2 + cnt] = Dtype(j);
          mask[index] = Dtype(cnt++);
      }
  }
  else if (prev_mask == NULL && label != NULL)
  {
    for (int i = 0; i < bottom[0]->height(); i++)
      for (int j = 0; j < bottom[0]->width(); j++)
      {
        const int index = i * bottom[0]->width() + j;
        mask[index] = Dtype(-1);
        //has label
        const int label_value = static_cast<int>(label[index]);
        if (label_value != ignore_label_)
        {
          //not ignore label
          if (prob[label_value * spatial_dim + index] > threshold_)
            continue;
        }
        else
        {
          //ignore label
          bool flag = false;
          for (int k=0; k < channels; k++)
            if (prob[k * spatial_dim + index] > threshold_)
            {
              flag = true;
              break;
            }
          if (flag) continue;
        }
        //keep pixel (i, j)
        idx[1 + cnt] = Dtype(i);
        idx[spatial_dim + 2 + cnt] = Dtype(j);
        mask[index] = Dtype(cnt++);
      }
  }
  else if (prev_mask == NULL && label == NULL)
  {
    for (int i = 0; i < bottom[0]->height(); i++)
        for (int j = 0; j < bottom[0]->width(); j++)
        {
          const int index = i * bottom[0]->width() + j;
          mask[index] = Dtype(-1);
          //get max probability
          bool flag = false;
          for (int k=0; k < channels; k++)
            if (prob[k * spatial_dim + index] > threshold_)
            {
              flag = true;
              break;
            }
          if (flag) continue;
          //keep pixel (i, j)
          idx[1 + cnt] = Dtype(i);
          idx[spatial_dim + 2 + cnt] = Dtype(j);
          mask[index] = Dtype(cnt++);
      }
  }

  /*
  for (int i = 0; i < bottom[0]->height(); i++)
    for (int j = 0; j < bottom[0]->width(); j++)
    {
      const int index = i * bottom[0]->width() + j;
      mask[index] = Dtype(-1);
      
      //prev mask
      if (prev_mask != NULL && static_cast<int>(prev_mask[index]) == -1)
        continue;

      //get max probability
      int max_prob = 0;
      for (int k=1; k < channels; k++)
        if (prob[k * spatial_dim + index] > prob[max_prob * spatial_dim + index])
          max_prob = k;

      
      if (label != NULL)
      {
        //has label
        const int label_value = static_cast<int>(label[index]);
        if (label_value != ignore_label_)
        {
          //not ignore label
          if (prob[label_value * spatial_dim + index] > threshold_)
            continue;
        }
        else
        {
          //ignore label
          if (prob[max_prob * spatial_dim + index] > threshold_)
            continue;
        }
      }
      else
      {
        //no label
        if (prob[max_prob * spatial_dim + index] > threshold_)
          continue;
      }

      //keep pixel (i, j)
      idx[cnt] = Dtype(i);
      idx[spatial_dim + 1 + cnt] = Dtype(j);
      mask[index] = Dtype(cnt++);
  }*/
  idx[0] = Dtype(cnt);
  idx[spatial_dim + 1] = Dtype(cnt);
  if (top.size() > 2)
    top[2]->mutable_cpu_data()[0] = Dtype(cnt);
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
