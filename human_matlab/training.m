clear; clc;

if exist('../matlab/+caffe', 'dir')
  addpath('../matlab');
end;
caffe.reset_all();


use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
file_solver = 'human_solver.prototxt';
caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from('lar23_sp_iter_199000.caffemodel');
%%
image_dir = '../data/renren';
image_train_dir = '../data/renren/train';
[image_list, label_list] = textread(fullfile(image_dir, 'train.txt'), '%s %s');




th = 225; tw = 225;

mean_value = ones(th, tw, 3, 'single');

mean_value(: ,:, 1) = 104.008;
mean_value(: ,:, 2) = 116.669;
mean_value(: ,:, 3) = 122.675;

iter_num = 0; idx = 1;
batch_size = 300;

while(iter_num < 200000)
    if (idx == 1)
        rank = randperm(length(image_list));
        image_list = image_list(rank);
        label_list = label_list(rank);
    end;
    tic;
    net_input = {[], []};
    for i = 1:batch_size
        img = imread(fullfile(image_train_dir, image_list{idx}));
        if (size(img, 3) == 1)
            img = repmat(img, [1, 1, 3]);
        end;
        label = imread(fullfile(image_train_dir, label_list{idx}));
        if (size(label, 3) ~= 1)
            label = max(label, [], 3);
        end;
        [img, label] = im_tf(img, label, th, tw);
        large_img = zeros(th, tw, 3, 'single');
        large_label = 255 * ones(th, tw, 1, 'single');
        
        st_h = 1 + floor((th - size(img , 1)) / 2);
        st_w = 1 + floor((tw - size(img , 2)) / 2);
        
        large_img(st_h:st_h+size(img , 1)-1, st_w:st_w+size(img , 2)-1, :) = single(img(:, :, [3 2 1])) - mean_value(1:size(img, 1), 1:size(img, 2), :);
        large_label(st_h:st_h+size(img , 1)-1, st_w:st_w+size(img , 2)-1, :) = single(label);
        
        net_input{1} = cat(4, net_input{1}, permute(large_img, [2, 1, 3]));
        net_input{2} = cat(4, net_input{2}, permute(large_label, [2, 1, 3]));
    end;
    toc;
    %caffe_solver.net.forward(net_input)
    caffe_solver.net.set_input_data(net_input);
    tic;
    caffe_solver.step(1);
    toc;
    if (mod(caffe_solver.iter(), 10) == 0)
        num =randi(size(net_input{1}, 4));
        subplot(221);
        temp = permute(net_input{1}(:, :, :, num), [2, 1, 3]);
        temp = uint8(temp + mean_value);
        imshow(temp(:, :,[3 2 1]));
        subplot(222);
        temp = permute(net_input{2}(:, :, :, num), [2, 1, 3]);
        imshow(single(temp));
        subplot(223);
        feat = caffe_solver.net.blobs('fc9_zoom').get_data();
        temp = permute(feat(:, :, :, num), [2, 1, 3]);
        imagesc(temp(:, :, 1));
        axis 'image';
        subplot(224);
        [~, max_label] = max(temp, [], 3);
        imshow(single(max_label - 1));
        drawnow;
    end;
end