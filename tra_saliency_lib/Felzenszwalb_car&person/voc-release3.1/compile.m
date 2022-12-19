mex -O resize.cc
mex -O dt.cc
mex -O features.cc

% use one of the following depending on your setup
% 1 is fastest, 3 is slowest 

% 1) multithreaded convolution using blas
%mex -O fconvblas.cc -lmwblas -O fconv.cc
% 2) mulththreaded convolution without blas
% mex -O fconvMT.cc -o fconv
% 3) basic convolution, very compatible
% mex -O fconv.cc -o fconv

mex -O fconv.cc