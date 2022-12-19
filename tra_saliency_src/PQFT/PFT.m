function saliencyMap = PFT(inImg)

%% Read image from file 
% inImg = im2double(rgb2gray(imread('yourImage.jpg')));
inImg = im2double(rgb2gray(inImg));
imgSize = size(inImg);
inImg = imresize(inImg, [64 64]);

%% Spectral Residual
myFFT = fft2(inImg); 
myPhase = angle(myFFT);
saliencyMap = abs(ifft2(exp(i*myPhase))).^2;

%% After Effect
saliencyMap = mat2gray(imresize(imfilter(saliencyMap, fspecial('gaussian', 8, 3)),imgSize));
% imshow(saliencyMap);