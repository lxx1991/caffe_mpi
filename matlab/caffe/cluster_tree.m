function cluster_tree()
% this is just to generate two layer tree layers
numPerSynset = 10; % num of examples selected for clustering per class
k = 5; % cluster centers

% read the synsets
synsets = {};
filename = 'synsets.txt';
fid = fopen(filename, 'r');
for i = 1 : 1000
    synsets{end+1,1} = fgetl(fid);
end
fclose(fid);
% read synsets word
words = {};
filename = 'synset_words.txt';
fid = fopen(filename, 'r');
for i = 1 : 1000
    words{end+1,1} = fgetl(fid);
end
fclose(fid);

% compute centers
feature_dir = 'features';
% feature_list = dir(feature_dir);
centers = zeros(1024,1000,'single');
% randomly select 100 features per class
all_features = zeros(1024,numPerSynset*1000,'single');
all_labels = zeros(1,numPerSynset*1000,'single');
for i = 1 : 1000
    load([feature_dir '/' synsets{i} '.mat']);
    centers(:,i) = mean(features, 2);
    pm_idx = randperm(size(features,2));
    all_features(:,(i-1)*numPerSynset+1:i*numPerSynset) = features(:,pm_idx(1:numPerSynset));
    all_labels(:,(i-1)*numPerSynset+1:i*numPerSynset) = i;
end

% clustering
[idx,c,sumd,d] = kmeans(all_features', k);
c_component = zeros(1000,k);
for i = 1 : k
    com = all_labels(idx==i);
    [a,~] = hist(com, 1:1000);
    c_component(:,i) = a' ./ numPerSynset;
end
max_threshold = min(max(c_component, [], 2), [], 1);
min_threshold = 1 / k;
fprintf('max_threshold: %f\n', max_threshold);
fprintf('min_threshold: %f\n', min_threshold);
c_hard = c_component > (max_threshold + min_threshold)/2;

c_words = cell(k,1);
for i = 1 : k
    temp = {};
    idx_hard = find(c_hard(:,i) == 1);
    for d = 1 : length(idx_hard)
        temp{end+1,1} = words{idx_hard(d)};
    end
    c_words{i} = temp;
end

% generating label files
% leaf node first
for i = 1 : k
    counter = 0;
    num_classes = sum(c_hard(:,i) == 1);
    filename = ['label' num2str(i) '_' num2str(num_classes)];
    fid = fopen(filename, 'w');
    for d = 1 : 1000
        if c_hard(d,i) == 0
            fprintf(fid,'%d\n',-1);
        else
            fprintf(fid,'%d\n',counter);
            counter = counter + 1;
        end
    end
    fclose(fid);
end

% root node
num_classes = k;
filename = ['label' num2str(0) '_' num2str(num_classes)];
fid = fopen(filename, 'w');
for d = 1 : 1000
    counter = sum(c_hard(d,:) == 1);
    fprintf(fid,'%d ',counter);
    assert(counter > 0 , 'something wrong here!!!------------------\n');
    for i = 1 : k
        if c_hard(d,i) == 1
            fprintf(fid,'%d ',i-1);
        end
    end
    fprintf(fid,'\n');
end
fclose(fid);