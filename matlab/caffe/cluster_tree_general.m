function cluster_tree_general()
% this is to generate arbitrary trees

numPerSynset = 200; % num of examples selected for clustering per class
k = [1000, 20, 5]; % cluster centers

% read the synsets
synsets = {};
filename = 'synsets.txt';
fid = fopen(filename, 'r');
for i = 1 : k(1)
    synsets{end+1,1} = fgetl(fid);
end
fclose(fid);
% read synsets word
words = {};
filename = 'synset_words.txt';
fid = fopen(filename, 'r');
for i = 1 : k(1)
    words{end+1,1} = fgetl(fid);
end
fclose(fid);

% compute centers
feature_dir = 'features';
% feature_list = dir(feature_dir);
centers = zeros(1024,k(1),'single');
% randomly select 100 features per class
all_features = zeros(1024,numPerSynset*k(1),'single');
%all_labels = zeros(1,numPerSynset*k(1),'single');
for i = 1 : k(1)
    load([feature_dir '/' synsets{i} '.mat']);
    centers(:,i) = mean(features, 2);
    pm_idx = randperm(size(features,2));
    all_features(:,(i-1)*numPerSynset+1:i*numPerSynset) = features(:,pm_idx(1:numPerSynset));
%    all_labels(:,(i-1)*numPerSynset+1:i*numPerSynset) = i;
end
%label_map = full(sparse(1:k(1), 1:k(1), 1));
leaf_map = full(sparse(1:k(1), 1:k(1), 1));
all_label_map = cell(length(k), 1);

for l = 2 : length(k)
    % generate labels and num_data per class for this layer
    all_labels = zeros(numPerSynset*k(1), 1);
    for i = 1 : k(1)
        i_idx = find(leaf_map(i,:) == 1);
        for p = 1 : numPerSynset 
            i_rand = randi(length(i_idx));
            all_labels((i-1)*numPerSynset+p) = i_idx(i_rand);
        end
    end
    bottom_classes = max(all_labels);
    assert(bottom_classes == size(leaf_map,2), 'data to be clustered mismatch with labels');
    numPerClass = zeros(bottom_classes,1);
    for c = 1 : bottom_classes
        numPerClass(c) = sum(all_labels == c);
    end
    
    % clustering
    [idx,c,sumd,d] = kmeans(all_features', k(l));
    c_component = zeros(k(l-1),k(l));
    for i = 1 : k(l)
        com = all_labels(idx==i);
        [a,~] = hist(com, 1:bottom_classes);
        c_component(:,i) = a';
    end
    c_component = bsxfun(@rdivide, c_component, numPerClass);
    
    max_threshold = min(max(c_component, [], 2), [], 1);
    min_threshold = 1 / k(l);
    fprintf('max_threshold: %f\n', max_threshold);
    fprintf('min_threshold: %f\n', min_threshold);
    label_map = c_component > (max_threshold + min_threshold)/2;
    all_label_map{l-1} = label_map;
    
    c_words = cell(k(l),1);
    for i = 1 : k(l)
        temp = {};
        idx_hard = find(label_map(:,i) == 1);
        for d = 1 : length(idx_hard)
            temp{end+1,1} = words{idx_hard(d)};
        end
        c_words{i} = temp;
    end
    
    % generating label filess
    assert(all(sum(label_map, 2) > 0), 'no label for a example -------\n');
    for i = 1 : k(l)
        k_label = label_map(:,i);
        node_classes = sum(k_label);
        temp = leaf_map(:,k_label);
        
        filename = ['layer' num2str(length(k)+2-l) '_label' num2str(i) '_' num2str(node_classes)];
        fid = fopen(filename, 'w');
        for d = 1 : k(1)
            fprintf(fid, '%d', sum(temp(d,:)));
            c_idx = find(temp(d,:) == 1);
            for t = 1 : length(c_idx)
                fprintf(fid, ' %d', c_idx(t)-1);
            end
            fprintf(fid,'\n');
        end
        %assert(counter == node_classes, 'classification num wrong\n');
        fclose(fid);
    end
    
    % update leaf_map to the new layer
    new_leaf_map = zeros(k(1), size(label_map,2));
    for d = 1 : k(1)
       temp = label_map(leaf_map(d,:) == 1, :);
       new_leaf_map(d,:) = double(sum(temp,1) > 0);
    end
    leaf_map = new_leaf_map;
end

% root node
all_labels = zeros(1, numPerSynset*k(1));
for i = 1 : k(1)
    i_idx = find(leaf_map(i,:) == 1);
    for p = 1 : numPerSynset 
        i_rand = randi(length(i_idx));
        all_labels((i-1)*numPerSynset+p) = i_idx(i_rand);
    end
end
bottom_classes = max(all_labels);
assert(bottom_classes == size(leaf_map,2), 'data to be clustered mismatch with labels');
numPerClass = zeros(bottom_classes,1);
for c = 1 : bottom_classes
    numPerClass(c) = sum(all_labels == c);
end

% clustering for root is just all ones
label_map = ones(bottom_classes, 1);
label_map = label_map > 0;
all_label_map{end} = label_map;

filename = ['layer1_label1_' num2str(k(end))];
fid = fopen(filename, 'w');
assert(all(sum(label_map, 2) > 0), 'no label for a example -------\n');
k_label = label_map(:,1);
temp = leaf_map(:,k_label);

for d = 1 : k(1)
    fprintf(fid, '%d', sum(temp(d,:)));
    c_idx = find(temp(d,:) == 1);
    for t = 1 : length(c_idx)
        fprintf(fid, ' %d', c_idx(t)-1);
    end
    fprintf(fid,'\n');
end
%assert(counter == node_classes, 'classification num wrong\n');
fclose(fid);


% use the connectivity matrix(label_map) to decide the final tree
tree = cell(length(k),1);
% root node : layer 1 has only one node
tree{1} = [1];
pre_nodes = [1];
pre_node_num = length(pre_nodes);
for l = length(k): -1 : 2
    label_map = all_label_map{l};
    temp_label_map = label_map(:,pre_nodes);
    node_num = sum(temp_label_map(:));
    nodes = zeros(node_num, 1);
    now = 1;
    for n = 1 : pre_node_num
        this_label = label_map(:,n);
        nodes(now:now+sum(this_label)-1) = find(this_label > 0);
        now = now + sum(this_label);
    end
    assert(node_num == now-1, 'tree node generation wrong\n');
    tree{length(k)+2-l} = nodes;
    pre_nodes = nodes;
    pre_node_num = length(pre_nodes);
end
save('tree','tree');

