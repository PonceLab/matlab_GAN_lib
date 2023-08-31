% First Time Setup on New Machines: 
%    download the python code from https://github.com/huggingface/pytorch-pretrained-BigGAN.git
%    download these weights and definitions and put them in savedir
%    specified in the class definition below. You can add your computer's
%    name to that 
        % resolved_config_file = "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json";
        % resolved_model_file = "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin";
% 
% Setup Python Env For Matlab and PyTorch to Get Along: 
%   setup the python env that have pytorch with version < 1.4.0 in it (1.3.1 and 1.1.0 have been proved)  
%   check which conda env is good by 
%    `conda activate xxx` 
%    `conda list`  and
%   copy the directory [envpath] for the proper env list in `conda env list` to the
%   `pyenv` command, e.g. 
%     `pyenv('Version','[envpath]\python.exe')` 
%   and run this FIRST when you start matlab.(usually only need to run once,
%   then matlab remember your environment. 
% 
% Environments that have been tested to work are these. 
%   on Binxu home `pyenv('Version','C:\ProgramData\Anaconda3\envs\tf-torch\python.exe')` 
%   on Office 3 `pyenv("Version","C:\Users\ponce\.conda\envs\caffe36\python.exe")`
%   on ML2a `pyenv("Version", "C:\Anaconda3\envs\torch\python.exe");`
%   on ML2b `pyenv("Version", "C:\Users\Ponce lab\.conda\envs\torch\python.exe")`
%     ML2c  `pyenv("Version", "C:\Users\ponce\.conda\envs\torch\python.exe")
%    
% Note, sometimes import numpy and torch can fail, then we need to add the
% path of binary of the Library to the PATH env variable. E.g.
% This is in `[envpath]\Library\bin` 
% 
%   setenv('path',['C:\Anaconda3\envs\torch\Library\bin;', getenv('path')]);
%   % WHEN IMPORT FAILS, RUN THIS LINE ABOVE
% 
% This add to path should be run each time. So I add it to init code
% **Note**: as the path is appended each time, the path list will keep getting
% longer if matlab is not restarted. THat can cause import error if the
% path get long enough and get rid of some important ones. 
%   
%   Binxu July.20th, 2020. Updated Oct. 9th
classdef torchBigGAN
   % Usage: 
   % Visualizing a certian class 
   %  BGAN = torchBigGAN("biggan-deep-256");
   %  matimgs = G.visualize_class(0.6*randn(5,128),729);figure;montage(matimgs)
   % 
   % Visualizing 256d latent code  
   %  BGAN = torchBigGAN("biggan-deep-256");
   %  truncnorm = truncate(makedist("Normal"),-2,2);
   %  matimgs = G.visualize_latent([truncnorm.random(5,128), randn(5,128)*0.06]); figure;montage(matimgs)
   % 
   % Fix part of the code and Visualizing the other half
   %  G = G.select_space("class", noise_vec);
   %  G.visualize(0.06 * randn(5,128))
   properties
       BGAN
       Embeddings
       Generator
       space % a variable preset to specify which space to use in visualize
       fix_noise_vec % internal state that can be fix to visualize the other half of space. 
       fix_class_vec
   end
   methods
   function G = torchBigGAN(modelname)
       if nargin == 0
           modelname = "biggan-deep-256";
       end
       switch getenv('COMPUTERNAME')
           case 'DESKTOP-9DDE2RH' % Office 3 Binxu's 
            savedir = "C:\Users\ponce\.pytorch_pretrained_biggan";
           case 'DESKTOP-MENSD6S' % Binxu's home work station
            savedir = "C:\Users\binxu\.pytorch_pretrained_biggan";
           case 'PONCELAB-ML2A' % MLa machine 
            savedir = "C:\Users\Poncelab-ML2a\Documents\Python\pytorch-pretrained-BigGAN\weights";
            setenv('path',['C:\Anaconda3\envs\torch\Library\bin;', getenv('path')]); % this gives the path to the dll and binary files. Or import will fail.
            % WHEN IMPORT FAILS, RUN THIS LINE.
            % pyenv("Version", "C:\Anaconda3\envs\torch\python.exe"); %
           case 'PONCELAB-ML2B' % MLb machine 
            savedir = "C:\Users\Ponce lab\Documents\Python\pytorch-pretrained-BigGAN\weights";
            setenv('path',['C:\Users\Ponce lab\.conda\envs\torch\Library\bin;', getenv('path')]); % this gives the path to the dll and binary files. Or import will fail.
            % WHEN IMPORT FAILS, RUN THIS LINE to RESET ENVIRONMENT
            % pyenv("Version", "C:\Users\Ponce lab\.conda\envs\torch\python.exe"); %
           case 'PONCELAB-ML2C' % MLc machine 
            savedir = "C:\Users\ponce\Documents\Python\pytorch-pretrained-BigGAN\weights";
           otherwise
            savedir = "C:\Users\Poncelab-ML2a\Documents\Python\pytorch-pretrained-BigGAN\weights";
        end
       % install the torch 1.3.x and the biggan package like below.
        py.importlib.import_module("torch");
        py.importlib.import_module("numpy");
        py.importlib.import_module('pytorch_pretrained_biggan');
        
       cfg = py.pytorch_pretrained_biggan.BigGANConfig();
       cfg = cfg.from_json_file(fullfile(savedir,compose("%s-config.json",modelname)));
       G.BGAN = py.pytorch_pretrained_biggan.BigGAN(cfg);
       G.BGAN.load_state_dict(py.torch.load(fullfile(savedir,compose("%s-pytorch_model.bin",modelname))));
       G.BGAN.to('cuda');G.BGAN.eval();
       py.torch.set_grad_enabled(false);
       tmp = py.list(G.BGAN.named_children);
       G.Embeddings = tmp{1}{2};
       G.Generator = tmp{2}{2};
   end
   function G = select_space(G, space, setting)
       % Set the space to configure the `visualize` function. Interface to
       % ML2 experiments where we want to preset the space at the start of
       % exp. For in silico evolution, also use this to configure which
       % space to optimize in. 
       % Note this function changes inner variables of G, so need to assign it back to G.
       % 
       % Default settings are 
       % G=G.select_space() % "noise" space, 
       % G=G.select_space("noise") % "noise" space, in golden fish class. 
       % G=G.select_space("noise", 374) % "noise" space, with class vector
       %                                 %  fixed as macaque 374 class
       % G=G.select_space("noise", rand(1000)) % "noise" space, with class vector
       %                      %  fixed as a projection of a 1000d vector
       % G=G.select_space("noise", 0.06*randn(128)) % "noise" space, with class vector
       %            %  fixed as a random 128d vector with proper norm
       % G=G.select_space("class") % "class" space, with noise vector
       %            %  fixed as a random 128d vector with proper norm
       % G=G.select_space("class", scaling) % "class" space, with noise vector
       %            %  fixed as a random 128d vector with proper norm * scaling
       % G=G.select_space("class", truncnorm.random(1,128)) % "class" space, with noise vector
       %            %  fixed as a random 128d vector sampled from truncated normal
       % 
       if nargin == 1
           G.space = "noise";
       else
       
       if contains(space, "noise")
           EmbedVects_mat = get_embedding(G);
           G.space = "noise";
           if nargin == 2 % default to the goldfish class
               setting = 2;
           end
           if numel(setting) == 1 % setting is a class id
               G.fix_class_vec = EmbedVects_mat(:,int32(setting))';
           elseif numel(setting) == 128 % setting is an 128D hidden vect
               G.fix_class_vec = setting;
           elseif numel(setting) == 1000 % setting is an one hot vector
               G.fix_class_vec = reshape(setting, 1, []) * EmbedVects_mat';
           else
               error("Second argument not understood...")
           end
           G.fix_noise_vec = nan(1,128); % set the other half of the code to nan
       elseif contains(space, "class")
           truncnorm = truncate(makedist("Normal"),-2,2);
           G.space = "class";
           if nargin == 2
               setting = 0.7;
           end
           if numel(setting) == 1 % setting is an scaler specifying **norm** of latent code
               G.fix_noise_vec = truncnorm.random([1,128]) * setting;
           elseif numel(setting) == 128 % setting is an 128D hidden vect
               G.fix_noise_vec = setting;
           else
               if isempty(setting)
                   error('The setting cannot be empty for BigGAN evolutions')
               else
               error("Second argument not understood...")
               end
           end
           G.fix_class_vec = nan(1,128); % set the other half of the code to nan
       else
           G.space = "all";
           G.fix_noise_vec = nan(1,128);
           G.fix_class_vec = nan(1,128);
       end
       end
   end
   
   %% sample_noise: function description
   function [codes] = sample_noise(G, num, space)
      if nargin<3, space="all"; end
      if strcmp(space,"all")
        truncnorm = truncate(makedist("Normal"),-2,2);
        noisevecs = truncnorm.random([num,128]);
        EmbedVects_mat = get_embedding(G);
        clsidx = randsample(1000,num);
        classvecs = EmbedVects_mat(:,clsidx)';
        codes = [noisevecs, classvecs];
      end
   end

   function matimgs = visualize(G, code)
       % interface with generate integrated code, cmp to FC6GAN
       switch G.space % depending on the space concatenate the hidden vectors in certain way
           case "class"
               code_cat = cat(2, repmat(G.fix_noise_vec, size(code, 1), 1), code);
               matimgs = G.visualize_latent(code_cat);
           case "noise"
               code_cat = cat(2, code, repmat(G.fix_class_vec, size(code, 1), 1));
               matimgs = G.visualize_latent(code_cat);
           case "all"
               matimgs = G.visualize_latent(code);
       end
   end
   
   function matimg = visualize_codes(G, noise, onehot, truncation)
       if nargin == 3, truncation=0.7;end
       tic
       imgs = G.BGAN(py.torch.tensor(py.numpy.array(noise)).view(int32(-1),int32(128)).float().cuda(),...
            py.torch.tensor(py.numpy.array(onehot)).view(int32(-1),int32(1000)).float().cuda(),truncation);
       toc
       matimg = imgs.detach.cpu().numpy().single;
       matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
   end
   
   function matimg = visualize_class(G, noise, classn, truncation)
       if nargin == 3, truncation=0.7;end
       onehot = zeros(size(noise,1),1000); onehot(:,classn)=1;
       tic
       imgs = G.BGAN(py.torch.tensor(py.numpy.array(noise)).view(int32(-1),int32(128)).float().cuda(),...
            py.torch.tensor(py.numpy.array(onehot)).view(int32(-1),int32(1000)).float().cuda(),truncation);
       toc
       matimg = imgs.detach.cpu().numpy().single;
       matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
   end
   
   function matimgs = visualize_latent(G, latent, truncation)
       if nargin == 2, truncation=0.7;end
       batchsize = 12;samplen = size(latent,1);csr = 1;
       tic
       matimgs = [];
       while csr <= samplen
       cnd = min(samplen,csr+batchsize);
       imgs = G.Generator(py.torch.tensor(py.numpy.array(latent(csr:cnd,:))).view(int32(-1),int32(256)).float().cuda(), truncation);
       matimg = imgs.detach.cpu().numpy().single;
       matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
       matimgs = cat(4, matimgs, matimg);
       csr = cnd + 1;
       end
       toc
   end

   function frame_cell = visualize_movie(G, latent_col, truncation)
       if nargin == 2, truncation=0.7;end
       batchsize = 10; movieN = numel(latent_col); 
       frameN = cellfun(@(code)size(code, 1), latent_col);
       samplen = sum(frameN); 
       latent_arr = cat(1, latent_col{:});
       % Generate latent code using the current space
       matimgs = G.visualize(latent_arr);
       % sort the frames into cells corresponding to each movie. 
       frame_cell = arrayfun(@(iMv) matimgs(:,:,:,sum(frameN(1:iMv-1))+1:sum(frameN(1:iMv))), 1:movieN, "Uni", 0); 
   end
   
   function EmbedVects_mat = get_embedding(G)
       EmbedVects = py.list(G.Embeddings.parameters());
       EmbedVects_mat = EmbedVects{1}.data.cpu().numpy().double;
   end
   end
end
