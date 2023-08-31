function out = pixelshuffle(input, upscale)
if nargin == 1
    upscale = 2;
end
H = size(input,1); W = size(input,2);
Ch = size(input,3) / upscale.^2;
B = size(input,4);
input_fact = reshape(input,[H,W,upscale,upscale,Ch,B]);
out = reshape(permute(input_fact, [4,1,3,2,5,6]), [H * upscale,W * upscale,Ch,B]);
end
% 
%%
% reshape(1:36,[1,9,2,2])
%%
% % input = permute(reshape(1:36,[3,3,4]),[2,1,3,4]);
% % fact_ch_input = reshape(input, [1,1,2,2,3,3]);
% % output = reshape(permute(fact_ch_input, [1,2,3,5,4,6]),[1,1,6,6]);
% %%
% input = permute(reshape(1:36,[3,3,4]),[2,1,3,4]);
% input_fact = reshape(input,[3,3,2,2,1]);
% out = reshape(permute(input_fact, [4,1,3,2]), [6,6,1,1]);
% %%
% upscale = 2;
% H = size(input,1); W = size(input,2);
% Ch = size(input,3) / upscale.^2;
% B = size(input,4);
% input_fact = reshape(input,[H,W,upscale,upscale,Ch,B]);
% out = reshape(permute(input_fact, [4,1,3,2,5,6]), [H * upscale,W * upscale,Ch,B]);
% %%
% input = permute(reshape(1:36,[3,3,4]),[2,1,3,4]);
% out = pixelshuffle(input);
% %%