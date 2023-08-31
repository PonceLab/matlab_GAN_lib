%% Demo code to import caffe into matlab and then compare result with python
%% Import GAN into matlab
% GANpath = "D:\Generator_DB_Windows\nets\upconv\fc6";
GANpath = "E:\Monkey_Data\Generator_DB_Windows\nets\upconv\fc6";
protofile = fullfile(GANpath, 'generator.prototxt');
datafile = fullfile(GANpath, 'generator.caffemodel');
% classNames = {'0','1','2','3','4','5','6','7','8','9'};
GAN = importCaffeNetwork(protofile, datafile, 'InputSize', [1, 1, 4096]);
Layers = GAN.Layers;
%%
I = GAN.forward(Z); % doesn't work 
%%
save('matlabGANfc6.mat','LinNet','DeconvNet','BGRMean')
%%
LinNet=dlnetwork(layerGraph(Layers(1:7)));
hiddeninputlayer = imageInputLayer([4,4,256],'Name','hiddeninput','Normalization','none');
DeconvNet=dlnetwork(layerGraph([hiddeninputlayer; Layers(9:25)]));
BGRMean = reshape([123.0, 117.0, 104.0],[1,1,3,1]);
%%
Z = dlarray(5*randn(1,1,4096,10), 'SSCB');
hiddenout = LinNet.predict(Z);
hiddenout_r = dlarray(hiddenout.reshape(4,4,256,[]),"SSCB");
out = DeconvNet.predict(hiddenout_r);
imgs = extractdata(out(:,:,:,:));
imgs = uint8(clip(imgs + BGRMean, 0, 255));
imgs = imgs(:,:,[3,2,1],:);
%%
basefolder = "N:\Stimuli\2019-06-Evolutions\beto-191212a\backup_12_12_2019_10_47_39";
load(fullfile(basefolder,"block030_thread000_code.mat"))
%
BGRMean = reshape([104.0, 117.0, 123.0],[1,1,3,1]);
tic
Z = dlarray(reshape(codes',1,1,4096,size(codes,1)), 'SSCB');
hiddenout = LinNet.predict(Z);
hiddenout_r = dlarray(hiddenout.reshape(4,4,256,[]).permute([2,1,3,4]),"SSCB"); % Note to flip your hiddenoutput
out = DeconvNet.predict(hiddenout_r);
imgs = extractdata(out(:,:,:,:));
imgs = uint8(clip(imgs + BGRMean, 0, 255));
imgs = imgs(:,:,[3,2,1],:);
toc
%
figure(7);set(7,'position',[1          41        2560         963]);
subplot(121);cla;montage(imgs(:,:,:,:))
%% Check the images are the same in montage
cd N:\Stimuli\2019-06-Evolutions\beto-191212a\backup_12_12_2019_10_47_39
imgname = string(ls(fullfile(basefolder, "block030_thread000*.jpg")));
subplot(122);cla;montage(imgname)
%%
code_id = 25;
img_py = imread(imgname(code_id));
img_mat = imgs(:,:,[3,2,1],code_id);
figure(8);
subplot(121);imshow(img_mat)
subplot(122);imshow(img_py)
%% 
residue = int16(img_mat) - int16(img_py);
figure(9);
subplot(121);imshow(uint8( residue));title("positive residue")
subplot(122);imshow(uint8(-residue));title("negative residue")
%% 
residue = int16(img_mat) - int16(img_py);
figure(10);
subplot(121);imshow(uint8( residue));title("positive residue")
subplot(122);imshow(uint8(-residue));title("negative residue")
%%
generator = FC6Generator('matlabGANfc6.mat');
imgs = generator.visualize(codes);
figure;montage(imgs)