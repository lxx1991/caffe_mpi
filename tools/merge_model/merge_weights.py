# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('mixture/fcn_res_101_flow_deploy.prototxt', 
                'mixture/fcn_res_101_iter_15000.caffemodel', 
                caffe.TEST)

net2 = caffe.Net('mixture/flow.prototxt', 
                'mixture/warping_iter_1000.caffemodel', 
                caffe.TEST)

params = ['flow_conv1', 'flow_conv2', 'flow_conv3', 'flow_conv3_1', 'flow_conv4', 'flow_conv4_1', 'flow_conv5', 'flow_conv5_1', 'flow_conv6', 'flow_conv6_1', 'Convolution1', 'deconv5', 'upsample_flow6to5', 'Convolution2', 'deconv4', 'upsample_flow5to4', 'Convolution3', 'deconv3', 'upsample_flow4to3', 'Convolution4', 'deconv2', 'upsample_flow3to2', 'Convolution5']
params2 = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6', 'conv6_1', 'Convolution1', 'deconv5', 'upsample_flow6to5', 'Convolution2', 'deconv4', 'upsample_flow5to4', 'Convolution3', 'deconv3', 'upsample_flow4to3', 'Convolution4', 'deconv2', 'upsample_flow3to2', 'Convolution5']
for i in range(0, len(params)):
	for j in range(0, len(net.params[params[i]])):
		net.params[params[i]][j].data.flat = net2.params[params2[i]][j].data.flat;


net.save('mixture/init_flow.caffemodel')