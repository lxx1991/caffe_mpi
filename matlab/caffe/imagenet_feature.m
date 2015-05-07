function imagenet_feature(path)
%path = '.';
path = '/home/sqiu/dataset/ILSVRC2012/train_resize_2';
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 


% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(1, '../../models/googlenet/deploy.prototxt', ...
      '../../models/googlenet/googlenet_train_iter_270000.caffemodel');
else
  matcaffe_init(1,'../../models/googlenet/deploy.prototxt', ...
      '../../models/googlenet/googlenet_train_iter_270000.caffemodel');
end

d = load('ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;
synsets = dir(path);
for c = 808 : length(synsets)
   if strcmp(synsets(c).name, '.') || strcmp(synsets(c).name, '..')
       continue;
   end
   class = synsets(c).name;
   fprintf('begin to extract %s\n', class);
   class_path = [path '/' class];
   
   % prepare image filename
   image_filename = dir(class_path);
   filelist = {};
   for i = 1 : length(image_filename)
       filename = [class_path '/' image_filename(i).name];
       if image_filename(i).isdir
           continue;
       end
       filelist{end+1} = filename; 
   end
   
   % get the features
   batch_num = ceil(length(filelist) / 64);
   features = zeros(1024, 64 * (batch_num - 1), 'single');
   for b = 1 : batch_num -1 
       input_data = {prepare_image(filelist, IMAGE_MEAN, (b-1)*64+1)};
       batch_feature = caffe('forward', input_data);
       batch_feature = squeeze(batch_feature{1});
       bsize = size(batch_feature,2);
       features(:, (b-1)*64+1:(b-1)*64+bsize) = batch_feature; 
   end
   
   feature_file = ['features/' class '.mat'];
   save(feature_file, 'features');
   fprintf('finished saving %s\n', class);
end

% ------------------------------------------------------------------------
function images = prepare_image(im_filelist, IMAGE_MEAN, start)
% ------------------------------------------------------------------------
IMAGE_DIM = 256;
CROPPED_DIM = 224;

to_end = min(start+64-1, length(im_filelist));
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, to_end-start+1, 'single');
for b = 1 : to_end-start+1
    % resize to fixed input size
    im = imread(im_filelist{b+start-1});
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    if size(im,3) == 1
        im = cat(3,im,im,im);
    end
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]) - IMAGE_MEAN;

    % just use the center to get the features
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    %{
    curr = 1;
    for i = indices
      for j = indices
        images(:, :, :, curr) = ...
            permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
        images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
        curr = curr + 1;
      end
    end
    %}
    center = floor(indices(2) / 2)+1;
    images(:,:,:,b) = ...
        permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
            [2 1 3]);
end
