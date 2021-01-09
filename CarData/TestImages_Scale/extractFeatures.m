function [imgOut] = extractFeatures(imgIn)
     [featursY,~] = imgradientxy(imresize((imgIn),[40,100]));
     features = [ featursY];
     flatImage = reshape(features,[],1);
     imgOut = normalize(flatImage);
end