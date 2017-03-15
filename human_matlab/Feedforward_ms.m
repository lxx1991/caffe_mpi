% This script contains pipeline of FCN + DenseCRF
clear; clc;
if exist('../matlab/+caffe', 'dir')
  addpath('../matlab');
end;
caffe.reset_all();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set Directory

data_set = 'renren';
list_name = 'test';

dir_data = fullfile('../data/', data_set);
dir_img = fullfile(dir_data, list_name);
file_list_name = fullfile(dir_data, [list_name, '.txt']);

 scales = 193 : 32 : 257;
% scales = [257+128]

dir_model = 'human_parsing';
file_model = fullfile(dir_model, 'fcn_res_101.model');
file_def_model = fullfile(dir_model, 'fcn_res_101.prototxt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization;

use_gpu = 0;

if use_gpu ~= -1
    caffe.set_mode_gpu();
    caffe.set_device(use_gpu);
else
    caffe.set_mode_cpu();
end

IMAGE_MEAN = zeros(1, 1, 3, 'single');
IMAGE_MEAN(: ,:, 1) = 104.008;
IMAGE_MEAN(: ,:, 2) = 116.669;
IMAGE_MEAN(: ,:, 3) = 122.675;
IMAGE_MEAN = imresize(IMAGE_MEAN, [scales(end) * 2, scales(end)], 'nearest');

[img_list, label_list]= textread(file_list_name, '%s %s');
num_img = length(img_list);


caffe.reset_all();

if use_gpu ~= -1
    caffe.set_mode_gpu();
    caffe.set_device(use_gpu);
else
    caffe.set_mode_cpu();
end

net = caffe.Net(file_def_model, file_model, 'test');
s1 = zeros(2, 1);
s2 = s1;

v = VideoReader('/DATA/video/segment/3/125_2.avi');
%v = VideoReader('/home/sensetime/to_lxx/segmenttest.MOV');
   
% for id_img = 1:num_img
tic;
while hasFrame(v)
    scores = [];
    im = readFrame(v);
%     im = imread(fullfile(dir_img, img_list{id_img}));
%     if (size(im, 3) == 1)
%         im = repmat(im, [1, 1, 3]);
%     end;
%     
%     label = imread(fullfile(dir_img, label_list{id_img}));
%     if (size(label, 3) ~= 1)
%         label = max(label, [], 3);
%     end;
    
    
    height_img = size(im, 1);
    width_img = size(im, 2);
        
    for scale = scales
        if (height_img > 2*width_img)
            image = imresize(im, [scale * 2, round(width_img * scale * 2 / height_img)]);
        else
            image = imresize(im, [round(height_img * scale / width_img), scale]);
        end;
        new_height_img = size(image, 1);
        new_width_img = size(image, 2);
        st_h = floor((scale * 2 - new_height_img) / 2);
        st_w = floor((scale - new_width_img) / 2);
        
        image = single(image(:, :, [3, 2, 1])) - IMAGE_MEAN(1:new_height_img, 1:new_width_img, :);
        input_image = zeros(scale * 2, scale, 3, 'single');
        input_image(st_h+1:st_h+new_height_img, st_w+1:st_w+new_width_img, :) = image;
        input_image = permute(input_image, [2, 1, 3]);
        input_data = {input_image};

        % init caffe network (spews logging info)
        net.reshape_as_input(input_data);
        score = net.forward(input_data);
        score = score{1};


        input_data = {input_image(end:-1:1, :, :)};
        score_f = net.forward(input_data);
        score = score + score_f{1}(end:-1:1, :, :);

        score = permute(score, [2, 1, 3]);
        score = imresize(score(st_h+1:st_h+new_height_img, st_w+1:st_w+new_width_img, :), [height_img, width_img]);
        
        if isempty(scores)
            scores = score;
        else
            scores = scores + score;
        end;
    end

    subplot(221);
    imshow(im);
    
%     subplot(222);
%     imshow(single(label));
    
    subplot(223);
    [~, max_label] = max(scores, [], 3);
    max_label = max_label - 1;
    imshow(single(max_label));
%     for j = 0:1
%         s1(j + 1) = s1(j + 1) + sum(max_label(:) == j & label(:) == j);
%         s2(j + 1) = s2(j + 1) + sum(max_label(:) == j) + sum(label(:) == j) - sum(max_label(:) == j & label(:) == j);
%     end;
%     
%     disp([s1(1) / s2(1), s1(2) / s2(2), (s1(1) / s2(1) + s1(2) / s2(2)) / 2]);
    drawnow;
end
toc;
caffe.reset_all();
