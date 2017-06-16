clc;
clear;
close all;
% addpath(genpath('functions'));

%% Setup Caffe
addpath('/home/sultan/caffe-nv/matlab/');

caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);
% caffe.set_mode_cpu();
net_weights = '20170227-182349-ebe7_epoch_10.0/snapshot_iter_23990.caffemodel';
net_model = '20170227-182349-ebe7_epoch_10.0/deploy.prototxt';

% net_weights = 'FineTunedHeadDetectorAug23/snapshot_iter_283166.caffemodel';
% net_model = 'FineTunedHeadDetectorAug23/deploy.prototxt';
net = caffe.Net(net_model, net_weights, 'test');

%%/media/sultan/D/GISTIC/tayyab/matlab_codes/data/wadiMakkah_images
data_path = 'images/';
files = dir([data_path '*.jpg']);
% files = files(8:end);
% files(1).name = 'mataf1.jpg';

rsize = 1.5;
step = 2;
min_hs = 2.0;
batch_size = 400;
patch_size = [224 224];

count = 1;
for i=1:1
    
    im = imread([data_path files(i).name]);
    if size(im, 3) == 1
        im(:,:,2) = im(:,:,1);
        im(:,:,3) = im(:,:,1);
    end
    
    if exist([data_path files(i).name(1:end-4) '_roi.mat'], 'file')
        load([data_path files(i).name(1:end-4) '_roi.mat']);
    else
        roi = true(size(im(:,:,1)));
    end
    
    load([data_path files(i).name(1:end-4) '_hsm.mat'])
    pad = ceil(rsize*max(hsm(:)));
    
    padded_im = padarray(im, [pad, pad], 0, 'both');
    padded_hsm = padarray(hsm, [pad, pad], 0, 'both');
    padded_roi = padarray(roi, [pad, pad], 0, 'both');    
    
    [x, y] = meshgrid(pad+1 :step: (size(padded_im,2)-pad), pad+1 :step: (size(padded_im,1)-pad));
    hsvec = padded_hsm(:, pad+1);    
    
    valid_x_mask = false(size(x));
    rects = zeros(numel(x), 4);
    for j=1:numel(x)

        chs = hsvec(y(j));
        flag = padded_roi(y(j), x(j));
        
        if chs > min_hs && flag == 1
            cx = x(j)- (chs*rsize/2);
            cy = y(j)- (chs*rsize/2);
            rects(j, :) = round([cx, cy, chs*rsize, chs*rsize]);
            valid_x_mask(j) = 1;
        end
    end
    
    response_mat = NaN(size(x));
    valid_x_ind = find(valid_x_mask);
    patches_to_process = length(valid_x_ind);
    tic;    
    for k=1:batch_size:patches_to_process
        c_idx = valid_x_ind(k : min(k+batch_size-1, patches_to_process));
        current_batch = zeros([patch_size 3 batch_size], 'single');
        for l=1:length(c_idx)
            cim = imcrop(padded_im, rects(c_idx(l), :));
            cim = imresize(cim, [224,224]);       % Size Normalization
            
%             cim = cim(:,:,[3,2,1]);               % RGB->BGR
%             cim(:,:,1) = cim(:,:,1) - 47.0;       % Mean subtraction
%             cim(:,:,2) = cim(:,:,2) - 52.0;
%             cim(:,:,3) = cim(:,:,3) - 57.0;
%             
%             cim = imresize(cim, [224,224]);       % Size Normalization
%             cim = permute(cim, [2,1,3]);          % Flip horozontal & vertical sides
%             
%             cim = single(cim);

            cim = cim(:,:,[3,2,1]);               % RGB->BGR

            cim = permute(cim, [2,1,3]);          % Flip horozontal & vertical sides
            cim = single(cim);

            cim(:,:,1) = cim(:,:,1) - 47.0;       % Mean subtraction
            cim(:,:,2) = cim(:,:,2) - 52.0;
            cim(:,:,3) = cim(:,:,3) - 57.0; 
            
            current_batch(:, :, :, l) = cim;
        end
        
        scores = net.forward({current_batch});
        hp = scores{1,1};
%         temp = process_batch(SVMModelPosteriorProb, current_batch);
        response_mat(c_idx) = hp(2, 1:length(c_idx));
        
        fprintf('Image %d of %d ... %.2f persent done.\n', i, length(files),  k/patches_to_process*100);
        count = count + 1
    end
    tt = toc
    
    response_mat( isnan(response_mat) ) = 0;
      
    if sum( size(hsm) == size(response_mat) ) == 2
        save([data_path files(i).name(1:end-4) '_response.mat'], 'response_mat');
    else
        save([data_path files(i).name(1:end-4) '_response_org.mat'], 'response_mat');
        response_mat = imresize(response_mat, size(hsm));
        
        response_mat(response_mat > 1) = 1;
        response_mat(response_mat < 0) = 0;
        save([data_path files(i).name(1:end-4) '_response.mat'], 'response_mat');
    end
    figure; imagesc(response_mat);
    refresh; pause(0.1);
end
