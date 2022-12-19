% clear
% clc

function saliencyMap = SR(inImg)

%% Read image from file 
% inImg = im2double(rgb2gray(imread('yourImage.jpg')));
inImg = im2double(rgb2gray(inImg));
imgSize = size(inImg);
inImg = imresize(inImg, 64/imgSize(2));

%% Spectral Residual
myFFT = fft2(inImg); 
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate'); 
saliencyMap = abs(ifft2(exp(mySpectralResidual + i*myPhase))).^2;

%% After Effect
saliencyMap = mat2gray(imresize(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)),imgSize));
% imshow(saliencyMap);