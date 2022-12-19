%%%%%%%%%%%%%%%%%%%%%%%%%%%%  HFT Saliency Computing  %%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% This computes the HFT saliency map for the input image                  %
% The hypercomplex FFT functions are provided by T. Ell[40]               %
%                                                                         %
% Jian Li. March,2011.                                                    %
%                                                                         %
% Email: lijian.nudt@gmail.com; lijian@cim.mcgill.ca                      %
% --You can use and distribute the code freely. If you use the HFT code   %
% --and the corresponding ROC code, please cite our PAMI paper:           %
% --Visual Saliency Based on Scale-Space Analysis in the Frequency Domain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SalMap = HFT(inImg1)
%% Initializtion 
rootInfo = what;root = rootInfo.path;root=strcat(root,'\functions\'); addpath(root)
param=callHFTParams;
%% Load image
inImg1 = double(inImg1);
[p1,p2,p3]=size(inImg1);
%% Resize image to 128*128 %You are encouraged to try other resolutions, or even learn a optimal image scale by training.
inImg1 = imresize(inImg1, [128, 128], 'bilinear');
%% Compute input feature maps
r = inImg1(:,:,1);g = inImg1(:,:,2);b = inImg1(:,:,3);   
I = mean(inImg1,3);I=max(max(r,g),b); % Results by "mean" are slightly different from that by "max"; Please try both of them for better results. We use "mean" for quantitative evaluation.
R = r-(g+b)/2;G = g-(r+b)/2;B = b-(r+g)/2;Y = (r+g)/2-abs(r-g)/2-b;Y(Y<0) = 0;
RG = double(R - G);BY =double(B - Y);
%% Compute the Hypercomplex representation
f = quaternion(0.25*RG, 0.25*BY, 0.5*I);  % the default weigthed for each feature map is 0.25 0.25 0.5
%% Compute the Scale space in frequency domain
[M,N]=size(r);S=MSQF(f,M,N);
%% Find the optimal scale
[H,W,Ch]=size(inImg1);sgm=W*param.SmoothingValue;
for k=1:8; 
      entro(k)=entropy1((S(:,:,k)));     %if run HFT, please use this line;
%     entro(k)=entropy2((S(:,:,k)));     %if run HFT(e), please use this line
end
entro_seq=sort(entro); c=find(entro==entro_seq(1));c=c(1);
SalMap=mat2gray(S(:,:,c));
for k=1:8
%     subplot(3,8,16+k)
    SalMap_k = imfilter(S(:,:,k), fspecial('gaussian',[round(4*sgm) round(4*sgm)],sgm));
%     imshow(SalMap_k,[])
%     if k==c
%     title(['k= ',num2str(k),'  E=',num2str(entro(k))],'fontsize',10,'Color','r')
%     else
%     title(['k= ',num2str(k),'  E=',num2str(entro(k))],'fontsize',10,'Color','b')
%     end
end
%--------------------------------------------------------------------------
%% Postprocessing
%-------------
%incorperate a border cut. A border cut could be employ to alleviate the
%problem caused by the border effect. However the unfairness will be
%introduced when make comparison. In our paper, the border cut is not used.
if param.openBorderCut == 1
SalMap=bordercut(SalMap,param.BorderCutValue);
end
%-------------
sgm=W*param.SmoothingValue;
SalMap = imfilter(SalMap, fspecial('gaussian',[round(4*sgm) round(4*sgm)],sgm));
SalMap = imresize(SalMap, [p1,p2], 'bilinear');
%-------------
%incorperate a global center bias
%However, we think that center bias has little significace, but only inrease
%ROC score. In our paper, the center bias is not incorperated.
if param.setCenterBias == 1
SalMap=CenterBias(SalMap,param.CenterBiasValue);
end
%-------------
SalMap=mat2gray(SalMap);

end