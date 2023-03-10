function saliencyMap = saliency(imagefile)
%
% saliencyMap = saliency(imagefile)
% This finds the saliency map of a given input image

% ----------------------------------------------------------------------
% Matlab tools for "Learning to Predict Where Humans Look" ICCV 2009
% Tilke Judd, Kristen Ehinger, Fredo Durand, Antonio Torralba
% 
% Copyright (c) 2010 Tilke Judd
% Distributed under the MIT License
% See MITlicense.txt file in the distribution folder.
% 
% Contact: Tilke Judd at <tjudd@csail.mit.edu>
% ----------------------------------------------------------------------

% load the image
img = imread(imagefile);
[w, h, c] = size(img);
dims = [200, 200];

% find all the necessary features for this image
% this will create a [w*h, numFeatures] array
% be sure to attach all the necessary libraries for these features to work.
% See the README.txt
FEATURES(:, 1:13) = findSubbandFeatures(img, dims);
FEATURES(:, 14:16) = findIttiFeatures(img, dims);
FEATURES(:, 17:27) = findColorFeatures(img, dims);
FEATURES(:, 28) = findTorralbaSaliency(img, dims);
FEATURES(:, 29) = findHorizonFeatures(img, dims);
FEATURES(:, 30:32) = findObjectFeatures(img, dims);
FEATURES(:, 33) = findDistToCenterFeatures(img, dims);
    
% load the model
% This model has been created to run with all the above 33 features
% If you'd like to run a model that has DIFFERENT features, you have to
% train your own model.  You can do this using the TrainAndTestModel code
% available on our website
% http://people.csail.mit.edu/tjudd/WherePeopleLook/Code/TrainAndTestModel.zip
load model

% whiten the feature data with the parameters from the model.
meanVec = model.whiteningParams(1, :);
stdVec = model.whiteningParams(2, :);
FEATURES=FEATURES-repmat(meanVec, [size(FEATURES, 1), 1]);
FEATURES=FEATURES./repmat(stdVec, [size(FEATURES, 1), 1]);

% find the saliency map given the features and the model
saliencyMap = (FEATURES*model.w(1:end-1)') + model.w(end);
saliencyMap = (saliencyMap-min(saliencyMap))/(max(saliencyMap)-min(saliencyMap));
saliencyMap = reshape(saliencyMap, dims);
saliencyMap = imresize(saliencyMap, [w, h]);

% show the results
if nargout==0
    figure;
    subplot(121); imshow(img); title('Original Image');
    subplot(122); imshow(saliencyMap); title('SaliencyMap');
end
