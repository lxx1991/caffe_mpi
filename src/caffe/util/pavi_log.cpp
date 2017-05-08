#include "caffe/util/pavi_log.hpp"

#include <algorithm>
#include <iterator>
#include <unistd.h>
#include <sstream>

#include "boost/thread.hpp"
#include "boost/asio.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/algorithm/string/replace.hpp"

namespace caffe {

	using boost::asio::ip::tcp;
	using namespace std;

	static string server_host_ = "", instance_id = "";

	string post_json(const string &host, const string &message)
	{
		boost::asio::io_service io_service;  
		boost::asio::ip::tcp::resolver resolver(io_service);  

		tcp::resolver::query query(host, "http");  
		tcp::resolver::iterator iter = resolver.resolve(query);
		tcp::socket socket(io_service);  
		boost::asio::connect(socket, iter);  

		boost::asio::streambuf request;  
		ostream request_stream(&request); 

		request_stream << "POST /log HTTP/1.0\r\n";  
		request_stream << "Host: " << host << "\r\n";  
		request_stream << "Content-Type: application/json\r\n";
		request_stream << "Content-Length: " << message.length() << "\r\n\r\n";
		request_stream << message;

		boost::asio::write(socket, request);  

		boost::asio::streambuf response;
		boost::asio::read_until(socket, response, "\r\n");

		// Check that response is OK.
		istream response_stream(&response);
		string http_version;
		response_stream >> http_version;
		unsigned int status_code;
		response_stream >> status_code;
		string status_message;
		getline(response_stream, status_message);
		if (!response_stream || http_version.substr(0, 5) != "HTTP/")
		{
		  LOG(ERROR) << "Invalid response";
		  return "";
		}
		if (status_code != 200)
		{
		  LOG(ERROR) << "Response returned with status code " << status_code;
		  return "";
		}

		// Read the response headers, which are terminated by a blank line.
		boost::asio::read_until(socket, response, "\r\n\r\n");

		// Process the response headers.
		string header;
		while (getline(response_stream, header) && header != "\r");

	    string response_str = "";
	    // Write whatever content we already have to output.
	    if (response.size() > 0)
	    {
	      ostringstream tmp;
	      tmp << &response;
	      response_str = response_str + tmp.str();
	    }

	    // Read until EOF, writing data to output as we go.
	    boost::system::error_code error;
	    while (boost::asio::read(socket, response,
	          boost::asio::transfer_at_least(1), error))
	    {
	      ostringstream tmp;
	      tmp << &response;
	      response_str = response_str + tmp.str();
	    }

		if (error != boost::asio::error::eof)
		{
		  LOG(ERROR) << "Recv Error!";
		  return "";
		}
		return response_str;
	}

	void pavi_init(const SolverParameter &solver_param, const string &net_name) {

		CHECK(solver_param.has_pavi_server_host());
	    server_host_ = solver_param.pavi_server_host();

		boost::property_tree::ptree tree;
		CHECK(solver_param.has_pavi_username());
		tree.put("username", solver_param.pavi_username());
		CHECK(solver_param.has_pavi_password());
		tree.put("password", solver_param.pavi_password());
		tree.put("instance_id", "");
		tree.put("session",  solver_param.session()); 
		tree.put("sessiontext", solver_param.DebugString());
		tree.put("model", net_name);
		tree.put("workdir", solver_param.workdir());
		char username[100];
		getlogin_r(username, 100);
		tree.put("device", string(username) + "@" + boost::asio::ip::host_name());
		string t = boost::posix_time::to_iso_extended_string(boost::posix_time::microsec_clock::universal_time());
		replace(t.begin(), t.end(), 'T', ' ');
		tree.put("time", t);
		stringstream ss;
		write_json(ss, tree);

		instance_id = post_json(server_host_, ss.str());
	}

	void pavi_send_log(const string &msg, const string &output_name, float output_value, const string &flow, const int iter)
	{
		if (instance_id.length() == 0)
			return ;

		boost::property_tree::ptree tree, output;
		//tree.put("lr", lr);
		tree.put("instance_id", instance_id);
		string t = boost::posix_time::to_iso_extended_string(boost::posix_time::microsec_clock::universal_time());
		boost::replace_all<string>(t, "T", " ");
		tree.put("time", t);
		tree.put("msg", msg);
		output.put(output_name, "output_value");
		tree.add_child("outputs",  output); 
		tree.put("flow", flow);
		tree.put("iter", "iter_value");
		stringstream ss;
		write_json(ss, tree);

		string message = ss.str();

		char buf[40];
		sprintf(buf, "%lf", output_value);
		boost::replace_all<string>(message, "\"output_value\"", string(buf));
		sprintf(buf, "%d", iter);
		boost::replace_all<string>(message, "\"iter_value\"", string(buf));

		boost::thread th(post_json, server_host_, string(message));
	}
}