#ifndef PAVI_LOG_HPP
#define PAVI_LOG_HPP

#include <iostream> 
#include <string>

#include "caffe/solver.hpp"
#include "caffe/common.hpp"

namespace caffe {

	using std::string;

	void pavi_init(const SolverParameter &solver_param, const string &net_name);


	void pavi_send_log(const string &msg, const string &output_name, float output_value, const string &flow, const int iter);
}


#endif  // PAVI_LOG_HPP