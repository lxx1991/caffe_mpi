clear; clc;

if exist('../matlab/+caffe', 'dir')
  addpath('../matlab');
end;

caffe.reset_all();

use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);

net = caffe.Net('human_parsing/fcn_res_101.model', 'human_parsing/fcn_res_101.prototxt', 'test');
%%
image_dir = '../data/renren';
image_train_dir = '../data/renren/test';
[image_list, label_list] = textread(fullfile(image_dir, 'test.txt'), '%s %s');

th = 512+128; tw = 512+128;

mean_value = ones(th, tw, 3, 'single');

mean_value(: ,:, 1) = 104;
mean_value(: ,:, 2) = 117;
mean_value(: ,:, 3) = 123;

net_input = {[]};
s1 = zeros(2, 1);
s2 = s1;
for i = 1:length(image_list)
    img = imread(fullfile(image_train_dir, image_list{i}));
    if (size(img, 3) == 1)
        img = repmat(img, [1, 1, 3]);
    end;
    label = imread(fullfile(image_train_dir, label_list{i}));
    if (size(label, 3) ~= 1)
        label = max(label, [], 3);
    end;
    [img, ~] = im_tf(img, label, th, tw);
    large_img = zeros(th, tw, 3, 'single');
    large_label = 255 * ones(th, tw, 1, 'single');

    st_h = 1 + floor((th - size(img , 1)) / 2);
    st_w = 1 + floor((tw - size(img , 2)) / 2);
    
    large_img(st_h:st_h+size(img , 1)-1, st_w:st_w+size(img , 2)-1, :) = single(img(:, :, [3 2 1])) - mean_value(1:size(img, 1), 1:size(img, 2), :);

    net_input{1} = permute(large_img, [2, 1, 3]);
    
    output = net.forward(net_input);
    
    subplot(221);
    %imshow(img);
    
    subplot(222);
    %imshow(single(label));
    
    subplot(223);
    output = permute(uint8(output{1}), [2, 1, 3]);
    output = output(st_h:st_h+size(img , 1)-1, st_w:st_w+size(img , 2)-1, :);
    output = imresize(output, [size(label, 1), size(label, 2)]);
    [~, max_label] = max(output, [], 3);
    max_label = max_label - 1;
    %imshow(single(max_label));
    
    for j = 0:1
        s1(j + 1) = s1(j + 1) + sum(max_label(:) == j & label(:) == j);
        s2(j + 1) = s2(j + 1) + sum(max_label(:) == j) + sum(label(:) == j) - sum(max_label(:) == j & label(:) == j);
    end;
    
    disp([s1(1) / s2(1), s1(2) / s2(2), (s1(1) / s2(1) + s1(2) / s2(2)) / 2]);
    
    %drawnow;
    %pause;
end;









