
function imageSaliency = Wu_ImageSaliencyComputing( imgName, blockSize, nComponent, smoothingSize)
        
    srcImg = imread(imgName);                                                
    imageSaliency = Real_Wu_ImageSaliencyComputing( srcImg, blockSize, smoothingSize, nComponent, ...
                                               1, 1, 1, 1);
                                               
return;                             



function imageSaliency = Real_Wu_ImageSaliencyComputing( image, blockSize, smoothingSize, nComponent,...
                                                    sigma1, sigma2, r1, r2)
    
    parameters = setParam( size( image, 1 ), size( image, 2 ), size( image, 3 ),...
                           blockSize, smoothingSize, nComponent, sigma1, sigma2, r1, r2 );
                    
    imgAftChngClrSpace = changeClrSpace( image, parameters );
    if isempty(imgAftChngClrSpace)
        disp('Fail');
        return;
    end
        
    allPatchData = getOneImgInfo( imgAftChngClrSpace, parameters );
    
    allPatchData.dataAftRdDim = doRdDim( double( allPatchData.clrDataMatrix ), parameters );
    
    saliencyMap = calOneImgSal( allPatchData, parameters );
    
    salMapAftCenBias = saliencyMap;
    
    imageSaliency = postProcessSalImg( salMapAftCenBias, parameters );
            
    clear salMapAftCenBias;
    clear saliencyMap;
    clear allPatchData;
    clear imgAftChngClrSpace;
    clear imgAftReArrange;
    clear parameters;    

return;


function parameters = setParam( imageHeight, imageWidth, imageChan, ...
                                blockSize, smoothingSize, nComponent,...
                                sigma1, sigma2, r1, r2)

    parameters.imageHeight = imageHeight;
    parameters.imageWidth = imageWidth;
    parameters.imageChannel = imageChan;
    parameters.winHeight = blockSize;
    parameters.winWidth = blockSize;
    parameters.nComponent = nComponent;
    parameters.smoothingSize = smoothingSize;
    parameters.sigma1 = sigma1;
    parameters.sigma2 = sigma2;
    parameters.r1 = r1;
    parameters.r2 = r2;
    
return;


function imgYCbCr = changeClrSpace( imgRGB, parameters )
    
    if parameters.imageChannel == 3
        imgYCbCr = rgb2ycbcr(imgRGB); 
    elseif parameters.imageChannel == 1
        imgYCbCr = imgRGB; 
    else
        imgYCbCr = [];     
    end
    
return;


function allPatchData=getOneImgInfo( img, parameters )

    allPatchData.hei = parameters.imageHeight;
    allPatchData.wid = parameters.imageWidth;
    allPatchData.chan = parameters.imageChannel;   
    allPatchData.winH = parameters.winHeight;
    allPatchData.winW = parameters.winWidth;
    allPatchData.nPixel = allPatchData.winH * allPatchData.winW;
    allPatchData.statePerH = floor( allPatchData.hei / allPatchData.winH ); 
    allPatchData.statePerW = floor( allPatchData.wid / allPatchData.winW ); 
    allPatchData.nState = allPatchData.statePerH * allPatchData.statePerW;    
    allPatchData.eachPatch = cell( allPatchData.nState, 1 );
    allPatchData.allCenterH = zeros( 1,allPatchData.nState );
    allPatchData.allCenterW = zeros( 1,allPatchData.nState );
    allPatchData.allPatchIndexH = zeros( 1,allPatchData.nState );
    allPatchData.allPatchIndexW = zeros( 1,allPatchData.nState );
    
    clrImg = double( img );          
    allPatchData = calEachNode( allPatchData, clrImg, parameters );
    allPatchData = changePatchClrValue( allPatchData ); 
    allPatchData = getClrDataMatrix( allPatchData ); 
    
    clear clrImg;
        
return;


function allPatchData = calEachNode( allPatchData, clrImg, parameters )

    for i = 1 : allPatchData.statePerH
        for j = 1 : allPatchData.statePerW
            uplfH = ( i - 1 ) * allPatchData.winH + 1;
            uplfW = ( j - 1 ) * allPatchData.winW + 1;
            uprgH = uplfH + allPatchData.winH - 1;
            uprgW = uplfW + allPatchData.winW - 1;
            patchIndex = j + ( i - 1 ) * allPatchData.statePerW;
            
            allPatchData.eachPatch{ patchIndex }.indByHei = i;
            allPatchData.eachPatch{ patchIndex }.indByWid = j;
            allPatchData.eachPatch{ patchIndex }.centerH = floor( ( uplfH + uprgH ) / 2 );
            allPatchData.eachPatch{ patchIndex }.centerW = floor( ( uplfW + uprgW ) / 2 );
            allPatchData.eachPatch{ patchIndex }.clrValue = clrImg( uplfH : uprgH, uplfW : uprgW,...
                                                            1 : parameters.imageChannel );
            allPatchData.allCenterH( patchIndex ) = floor( ( uplfH + uprgH ) / 2 );
            allPatchData.allCenterW( patchIndex ) = floor( ( uplfW + uprgW ) / 2 );                                           
            allPatchData.allPatchIndexH(patchIndex) = i;
            allPatchData.allPatchIndexW(patchIndex) = j;
        end
    end
    
return;


function allPatchData = changePatchClrValue( allPatchData )

    tmpSumMatrix = zeros( allPatchData.winH, allPatchData.winW, allPatchData.chan );    
    for i = 1 : allPatchData.nState
        tmpSumMatrix = tmpSumMatrix + allPatchData.eachPatch{ i }.clrValue;    
    end
    tmpSumMatrix = tmpSumMatrix / allPatchData.nState;

    for i = 1 : allPatchData.nState
        allPatchData.eachPatch{ i }.clrValue = abs( allPatchData.eachPatch{ i }.clrValue - ...
                                                    tmpSumMatrix );
    end 
    
    clear tmpSumMatrix;
    
return;


function allPatchData = getClrDataMatrix( allPatchData )

    allPatchData.clrDataMatrix = zeros(allPatchData.nPixel * allPatchData.chan, allPatchData.nState);
    
    tmp = zeros( allPatchData.winH, allPatchData.winW, allPatchData.chan ) ;
    for i = 1 : allPatchData.nState
        tmp = allPatchData.eachPatch{i}.clrValue;
        allPatchData.clrDataMatrix( :, i ) = tmp( : );
    end
    
    clear tmp;

return;


function dataAftRdDim = doRdDim( dataMatrix, parameters )

    dim = parameters.nComponent;
    tranpDataMatrix = dataMatrix';
    opts.disp = 0;
    [ tmpDataAftRdDim, D ] = eigs( tranpDataMatrix * tranpDataMatrix', dim, 'lm', opts ); 
    dataAftRdDim= tmpDataAftRdDim';
        
    clear D;
    clear tmpDataAftRdDim;
    clear tranpDataMatrix;
    
return;


function saliencyMap = calOneImgSal( allPatchData, parameters )

    salMapAftCenSurr = calCenSurr(allPatchData, parameters );
   
    if parameters.smoothingSize == 0
        saliencyMap = salMapAftCenSurr;
    else
        saliencyMap = imfilter( salMapAftCenSurr, fspecial('gaussian', ...
                                parameters.smoothingSize, parameters.smoothingSize ),...
                                'symmetric', 'conv');  
    end
       
    clear salMapAftCenSurr;
                        
return;


function salMapAftCenSurr = calCenSurr( allPatchData, parameters )
   
    sigma1 = parameters.sigma1;
    sigma2 = parameters.sigma2;
    r1 = parameters.r1;
    r2 = parameters.r2;

    hei = allPatchData.statePerH;
    wid = allPatchData.statePerW;
    maxSize = max( hei,wid );
    nState = hei * wid;
    
    tmpResult = double( zeros(1,nState) );
    allCenterH = allPatchData.allPatchIndexH;
    allCenterW = allPatchData.allPatchIndexW;
    dataAftRdDim = allPatchData.dataAftRdDim;
    
    oriAllCenterH = allCenterH;
    oriAllCenterW = allCenterW;
    oriHorMidPoint = oriAllCenterW(floor( length(oriAllCenterW)/2 ));
    oriVerMidPoint = oriAllCenterH(floor( length(oriAllCenterH)/2 ));
    
    horMidPoint = allCenterW(floor( length(allCenterW)/2 )); 
    verMidPoint = allCenterH(floor( length(allCenterH)/2 ));
    
    w1 = 0;
    w2 = zeros( 1,allPatchData.nState );
    dissimilarity = zeros( 1,allPatchData.nState );
    
    distMatrix = getdistMatrix( hei, wid );
    tmpMat = distMatrix';
    distMatInVec = tmpMat(:);

    tmpFac = sqrt((hei/2)^2 + (wid/2)^2);
    
    for i = 1 : nState                
        disInSpa = sqrt((allCenterH(i)-allCenterH).^2+(allCenterW(i)-allCenterW).^2);
        tmpRes = 1 ./ (1+disInSpa);
        dissimilarity = sum(abs(repmat(dataAftRdDim(:,i),1,nState)-dataAftRdDim));  
        tmpResult(i) = distMatInVec(i)*sum(tmpRes.*dissimilarity);
        
    end
    
    salMapAftCenSurr = ( reshape(tmpResult,wid,hei) )';
      
    clear dissimilarity;
    clear w2;
    clear w1;
    clear dataAftRdDim;
    clear tmpResult;
    
return;


function salMapAftCenBias = mulitiplyCentetBias( saliencyMap )

    distMatrix = findDistToCenterFeatures( saliencyMap );
    salMapAftCenBias = distMatrix.*mat2gray( saliencyMap );
    
    clear distMatrix;

return;


function distMatrix = findDistToCenterFeatures( img )

    [imgr, imgc, cc] = size( img );
    midpointx = floor( imgr/2 );
    midpointy = floor( imgc/2 );
    distMatrix = zeros( imgr, imgc );
    for x = 1 : imgr
        for y = 1 : imgc
            distMatrix( x, y ) = floor( sqrt( ( x-midpointx )^2 + ( y-midpointy )^2 ) );
        end
    end
    distMatrix = distMatrix / max( distMatrix( : ) );
    distMatrix= mat2gray( distMatrix );
    distMatrix = imadjust( distMatrix, [0 1], [1 0] );
   
return;


function imageSaliency = postProcessSalImg( salMapAftCenBias, parameters )  

    imageSaliency = imresize( mat2gray( salMapAftCenBias ),...
                              [ parameters.imageHeight parameters.imageWidth ] );

return;

function distMatrix = getdistMatrix( hei, wid )

    imgr = hei;
    imgc = wid;    
    midpointx = floor( imgr/2 );
    midpointy = floor( imgc/2 );
    distMatrix = zeros( imgr, imgc );
    for x = 1 : imgr
        for y = 1 : imgc
            distMatrix( x, y ) = floor( sqrt( ( x-midpointx )^2 + ( y-midpointy )^2 ) );
        end
    end
    distMatrix = distMatrix / max( distMatrix( : ) );
    distMatrix = 1 - distMatrix;  
    distMatrix = double(distMatrix);
   
return;
