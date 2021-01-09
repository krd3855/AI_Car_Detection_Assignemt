%% Author : Krishna Satish D S
clear all
clc
    load('Weights.mat');
    %% Read and Normalizing Data
    cd TestImages_Scale\
    directory = dir('*.pgm');
    names = {directory.name};
%% Reading data
 bw_data{100} = [];          %% Preallocation
 for i=1:length(names)
     d_img = imresize(imread(names{i}),[40 100]);
     f3 = extractFeatures(d_img);
     hog2 = f3;
     bw_data{i} = normalize(reshape(hog2,[],1));
 
 end
 cd ..
 correctClassified = 0;
 misclassifiedCount = 0;
  for i=1:(length(names))
            Input_Layer_Weight = (bw_data{i}'*Weight_Input_Hidden);  %% Wx --> Input layer to first hidden layer
            Input_Layer_Weight_Bias = Input_Layer_Weight + Weight_Bias_Hidden';  %% Wx + b
            Hidden_Layer_Input = sigmoid(Input_Layer_Weight_Bias);   %% Sigmoid Activation Function
            Hidden_Layer_temp = (Hidden_Layer_Input *  Weight_Hidden) + Weight_Bias_Output';  %% Hidden Layer Inputs multiplied with Hidden Layer Weights
            Final_Output = sigmoid(Hidden_Layer_temp)
            if(Final_Output > 0.2);
               correctClassified = correctClassified + 1;
            else
                misclassifiedCount = misclassifiedCount + 1;
            end
           
            
  end
X = categorical({'misclassifiedCount','correctClassified'});
X = reordercats(X,{'misclassifiedCount','correctClassified'});
Y = [misclassifiedCount correctClassified];
bar(X,Y)

