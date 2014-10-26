#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
#ifdef USE_MPI
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    string rank_str = static_cast<ostringstream*>(&(ostringstream()<<myrank))->str();
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.data_param().source(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
                           << this->layer_param_.data_param().source() << std::endl
                           << status.ToString();
#else
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.data_param().source(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
                           << this->layer_param_.data_param().source() << std::endl
                           << status.ToString();
#endif

    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
#ifdef USE_MPI
	int all_rank, my_rank;//, mpi_step_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &all_rank);
	unsigned int skip;
	if (this->layer_param_.data_param().rand_skip()) {
		skip = caffe_rng_rand() %
				this->layer_param_.data_param().rand_skip();
	}else{
		skip = 0;
	}
	LOG(INFO) << "Skipping first " << skip << " data points.";
	switch (this->layer_param_.data_param().backend()) {
		case DataParameter_DB_LEVELDB:
		//No idea how to set process skip for leveldb. it doesn't support db stats
			skip = this->layer_param_.data_param().mpi_skip_step()*my_rank;
			LOG(INFO)<<"mpi rank skipping "<< skip;
		break;
		case DataParameter_DB_LMDB:
		//Get the db size, split it into $(all_rank) parts by skiping corresponding items
//			MDB_stat stats;
//			mdb_env_stat(mdb_env_, &stats);
//			mpi_step_size = stats.ms_entries / all_rank;
//			skip += my_rank*mpi_step_size;
//			LOG(INFO)<<"mpi rank skipping "<< skip;

		break;
		default:
			LOG(FATAL) << "Unknown database backend";
	}
	while (skip-- > 0) {
		switch (this->layer_param_.data_param().backend()) {
			case DataParameter_DB_LEVELDB:
			iter_->Next();
			if (!iter_->Valid()) {
				iter_->SeekToFirst();
			}
			break;
			case DataParameter_DB_LMDB:
			if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
					!= MDB_SUCCESS) {
				CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
								MDB_FIRST), MDB_SUCCESS);
			}
			break;
			default:
			LOG(FATAL) << "Unknown database backend";
		}
	}
	MPI_Barrier(MPI_COMM_WORLD); // wait for other process to finish running to start location
#else
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
#endif

  //Set up mem bbox map
      if (this->layer_param_.data_param().has_mem_data_source()){
      	string key_name;
      	int coord[4];
      	int label;
      	std::ifstream infile(this->layer_param_.data_param().mem_data_source().c_str());
      	int cnt_ = 0;
      	while(infile>>key_name>>label>>coord[0]>>coord[1]>>coord[2]>>coord[3]){
      		this->bbox_data_[key_name] = vector<int>(coord, coord + sizeof(coord)/sizeof(int));
      		cnt_++;
      	}
      	LOG(INFO)<<"Pushed "<<cnt_<<" coord records";
      }

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

	if((this->layer_param_.top_size()>=3) && (!this->layer_param_.data_param().has_mem_data_source())) {
		LOG(ERROR)<<"To use bbox mask, please provide a bbox text file.";
	}
	// Reshape the top blob 3 to record bbox info
	// 4 numbers for bbox coordinates, 2 numbers for corresponding image size
	if (this->layer_param_.top_size()>=3){
		(*top)[2]->Reshape(this->layer_param_.data_param().batch_size(), 6, 1, 1);
	}


	if (this->layer_param_.data_param().has_mem_data_source()) {
		this->prefetch_bbox_mask_.Reshape(this->layer_param_.data_param().batch_size(), 6, 1, 1);
		this->prefetch_bbox_mask_.mutable_cpu_data();
		this->prefetch_aux_data_.insert( this->prefetch_aux_data_.begin(), &this->prefetch_bbox_mask_); // register the auxiliary prefetching data to the base class
	}
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  string key;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_bbox = NULL;
  vector<int> bbox;

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();
  if (this->layer_param_.top_size() == 3){
  	  top_bbox = this->prefetch_bbox_mask_.mutable_cpu_data();
    }
  int crop_size = this->layer_param_.transform_param().crop_size();
#ifndef USE_MPI
  for (int item_id = 0; item_id < batch_size; ++item_id) {
#else
  for (int item_id = batch_size * Caffe::mpi_self_rank() * (-1); item_id < batch_size * (Caffe::mpi_all_rank() - Caffe::mpi_self_rank()); ++item_id) {

	  bool do_read = (item_id>=0) && (item_id<batch_size);
	if(do_read){
#endif
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      key = iter_->key().ToString();
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      key = (char*)mdb_key_.mv_data;
//      LOG(INFO)<<"Read "<<item_id<<" "<<(char*)mdb_key_.mv_data;
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    //get the corresponding bounding box coordinates
	if (this->layer_param_.data_param().has_mem_data_source()){
		try{
			bbox = this->bbox_data_.at(key);
		}catch(const std::out_of_range& oor){
			bbox = vector<int>(4, 1);
		}
	}

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
		//Write the bbox coordinates to the prefetch buffer
	if (this->layer_param_.data_param().has_mem_data_source()) {
		for (int i = 0; i < 4; i++) {
			top_bbox[item_id * 6 + i] = bbox[i];
		}
		top_bbox[item_id * 6 + 4] = crop_size;
		top_bbox[item_id * 6 + 5] = crop_size;

	}
#ifdef USE_MPI
	}
	else{
//	    	LOG(INFO)<<" Skipped: "<<item_id<<" "<<(char*)mdb_key_.mv_data;
	    }
#endif
    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

  }
}

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
