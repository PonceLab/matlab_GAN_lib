% pe = pyenv('Version','C:\ProgramData\Anaconda3\envs\tf-torch\python.exe'); % Set up the python executable
py.importlib.import_module('torch');
py.importlib.import_module('lucent');
py.importlib.import_module('pytorch_pretrained_biggan');
import py.pytorch_pretrained_biggan.BigGAN
import py.pytorch_pretrained_biggan.BigGANConfig
% import py.pytorch_pretrained_biggan.convert_to_images_np
%%
% download these weights and definitions and put them somewhere.
% resolved_config_file = "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json";
% resolved_model_file = "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin";
cfg = py.pytorch_pretrained_biggan.BigGANConfig();
cfg = cfg.from_json_file("C:\Users\binxu\.pytorch_pretrained_biggan\biggan-deep-256-config.json");
BGAN = BigGAN(cfg);
BGAN.load_state_dict(py.torch.load("C:\Users\binxu\.pytorch_pretrained_biggan\biggan-deep-256-pytorch_model.bin"))
BGAN.to('cuda');BGAN.eval()
py.torch.set_grad_enabled(false)
% BGAN.from_pretrained("biggan-deep-128")
%%
classn = 602;690;579;%366;374;
noise = 0.7*randn(20,128);%randn(1,128);
onehot = zeros(20,1000);onehot(:,classn)=1;
tic
% py.torch.zeros(py.tuple([int32(1),int32(1000)]))%view(int32(-1),int32(128))
% py.torch.randn(py.tuple([int32(1),int32(128)]))%view(int32(-1),int32(1000))
img = BGAN(py.torch.tensor(py.numpy.array(noise)).view(int32(-1),int32(128)).float().cuda(),...
    py.torch.tensor(py.numpy.array(onehot)).view(int32(-1),int32(1000)).float().cuda(),0.7);
toc
matimg = img.detach.cpu().numpy().single;
matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
figure;
montage(matimg)
title(classn)
% model = BigGAN.from_pretrained('biggan-deep-256')
% model.to('cuda')
% 
% def BigGAN_render(class_vector, noise_vector, truncation):
%     if class_vector.shape[0] == 1:
%         class_vector = np.tile(class_vector, [noise_vector.shape[0], 1])
%     if noise_vector.shape[0] == 1:
%         noise_vector = np.tile(noise_vector, [class_vector.shape[0], 1])
%     class_vector = torch.from_numpy(class_vector.astype(np.float32)).to('cuda')
%     noise_vector = torch.from_numpy(noise_vector.astype(np.float32)).to('cuda')
%     with torch.no_grad():
%         output = model(noise_vector, class_vector, truncation)
%     imgs = convert_to_images(output.cpu())
%     return imgs