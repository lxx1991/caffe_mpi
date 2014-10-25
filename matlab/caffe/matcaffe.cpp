//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    CHECK(mxIsSingle(elem))
        << "MatCaffe require single-precision float point data";
    CHECK_EQ(mxGetNumberOfElements(elem), input_blobs[i]->count())
        << "MatCaffe input size does not match the input size of the network";
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}

static mxArray* do_get_feature_maps() {
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();

	  const vector<string>& layer_names = net_->layer_names();

	  // Step 1: count the number of layers with weights
	  int num_layers = 0;
	  int num_feature_maps = 0;
	  int copied_feature_maps = 0;
	  {
	    string prev_layer_name = "";
	    for (unsigned int i = 0; i < layers.size(); ++i) {
	      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
	      const vector<Blob<float>*>& top_vec = top_vecs[i];

//	      if (layer_blobs.size() == 0) {
//	        continue;
//	      }
//
	      if (layer_names[i] != prev_layer_name) {
	        prev_layer_name = layer_names[i];
	        num_layers++;
	        num_feature_maps += top_vec.size();
	      }
	    }
	  }

	  // Step 2: prepare output array of structures
	  mxArray* mx_layers;
	  {
	    const mwSize dims[2] = {num_layers, 1};
	    const char* fnames[3] = {"weights", "layer_names", "feature_maps"};
	    mx_layers = mxCreateStructArray(2, dims, 3, fnames);
	  }

	  // Step 3: copy weights into output

	  {

	    string prev_layer_name = "";
	    int mx_layer_index = 0;
	    for (unsigned int i = 0; i < layers.size(); ++i) {
	      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
	      const vector<Blob<float>*>& top_vec = top_vecs[i];
//	      if (layer_blobs.size() == 0) {
//	        continue;
//	      }

	      mxArray* mx_layer_cells = NULL;
	      mxArray* mx_feature_map_cells = NULL;
	      if (layer_names[i] != prev_layer_name) {
	        prev_layer_name = layer_names[i];
	        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
	        const mwSize fdims[2] = {static_cast<mwSize>(top_vec.size()), 1};
	        mx_layer_cells = mxCreateCellArray(2, dims);
	        mx_feature_map_cells = mxCreateCellArray(2, fdims);
	        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
	        mxSetField(mx_layers, mx_layer_index, "feature_maps", mx_feature_map_cells);
	        mxSetField(mx_layers, mx_layer_index, "layer_names",
	            mxCreateString(layer_names[i].c_str()));
	        mx_layer_index++;
	      }

	      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
	        // internally data is stored as (width, height, channels, num)
	        // where width is the fastest dimension
	        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
	            layer_blobs[j]->channels(), layer_blobs[j]->num()};

	        mxArray* mx_weights =
	          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
	        mxSetCell(mx_layer_cells, j, mx_weights);
	        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

	        switch (Caffe::mode()) {
	        case Caffe::CPU:
	          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
	              weights_ptr);
	          break;
	        case Caffe::GPU:
	          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
	              weights_ptr);
	          break;
	        default:
	          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	        }
	      }
			for (unsigned int j = 0; j < top_vec.size(); ++j) {
				// internally data is stored as (width, height, channels, num)
				// where width is the fastest dimension
				mwSize dims[4] = { top_vec[j]->width(),
						top_vec[j]->height(), top_vec[j]->channels(),
						top_vec[j]->num() };

				mxArray* mx_feature_map = mxCreateNumericArray(4, dims,
						mxSINGLE_CLASS, mxREAL);
				mxSetCell(mx_feature_map_cells, j, mx_feature_map);
				float* feature_map_ptr = reinterpret_cast<float*>(mxGetPr(
						mx_feature_map));

				switch (Caffe::mode()) {
				case Caffe::CPU:
					caffe_copy(top_vec[j]->count(),
							top_vec[j]->cpu_data(), feature_map_ptr);
					break;
				case Caffe::GPU:
					caffe_copy(top_vec[j]->count(),
							top_vec[j]->gpu_data(), feature_map_ptr);
					break;
				default:
					LOG(FATAL)<< "Unknown caffe mode: " << Caffe::mode();
				}
				copied_feature_maps ++;
			}

	    }

	  }
	  CHECK_EQ(copied_feature_maps, num_feature_maps);
	  return mx_layers;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void get_feature_maps(MEX_ARGS){
  plhs[0] = do_get_feature_maps();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
//  if (nrhs != 1) {
//    LOG(ERROR) << "Only given " << nrhs << " arguments";
//    mexErrMsgTxt("Wrong number of arguments");
//  }
  if (nrhs ==1){
	  plhs[0] = do_forward(prhs[0]);
	  return;
  }

  if (nrhs == 2){
	  plhs[0] = do_forward(prhs[0]);
	  double get_fm = mxGetScalar(prhs[1]);
	  if ((get_fm >=0.9)&&(nlhs ==2)){
		  plhs[1] = do_get_feature_maps();
	  }

  }

}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_backward(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "backward",           backward        },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  // added by alex
  { "get_feature_maps",   get_feature_maps},
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }
#ifdef USE_MPI
  MPI_Init(NULL, NULL);
#endif
  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
