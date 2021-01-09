%% Author : Krishna Satish D S

function BackPropAlgorithMomentum()
    %% Desired Outputs
    class_2 = [0 1]';
    %% Read and Normalizing Data
    cd TrainImages\
    directory = dir('*.pgm');
    names = {directory.name};
%% Reading data
count = 1;
 bw_data{1000} = [];          %% Preallocation
 for i=1:length(names)
     img = (imresize(imread(names{i}),[40 100]));
     f1 = extractFeatures(img);
     hog1 =  f1;
     bw_data{count} = normalize(reshape(hog1,[],1));

     count = count + 1;
 end
 cd ..
    %% Network parameters
    Number_Input = 4000;
    Number_Of_Hidden_Nodes = 550;
    Number_Of_Outputs = 1;
    Learning_Rate = 0.0009;
    Iteration=200;
    Alpha = 0.7;
    Weight_Input_Hidden =         rand(Number_Input,Number_Of_Hidden_Nodes , 'gpuArray') * 0.001;
    Prev_Weight_Input_Hidden =    Weight_Input_Hidden;
    Weight_Bias_Hidden =          rand(Number_Of_Hidden_Nodes,1, 'gpuArray') * 0.001;
    Prev_Weight_Bias_Hidden =     Weight_Bias_Hidden;
    Weight_Bias_Output =          rand(Number_Of_Outputs,1, 'gpuArray') * 0.001;
    Prev_Weight_Bias_Output =     Weight_Bias_Output;
    Weight_Hidden =               rand(Number_Of_Hidden_Nodes,Number_Of_Outputs, 'gpuArray') * 0.001;
    Prev_Weight_Hidden =          Weight_Hidden;
    error=                        zeros(Number_Of_Outputs,Iteration);
    %% Delta for Momentum Calculations
    Delta_Prev_Weight_Hidden =    (zeros(Number_Of_Hidden_Nodes,1, 'gpuArray'));
    Delta_Prev_Weight_Input_Hidden = (zeros(Number_Input,Number_Of_Hidden_Nodes, 'gpuArray'));
    Delta_Prev_Weight_Bias_Output  = (zeros(Number_Of_Outputs,1, 'gpuArray'));
    Delta_Prev_Weight_Bias_Hidden  = zeros(Number_Of_Hidden_Nodes,1, 'gpuArray');
    %% Shuffling Training Data 
    [Shuffled_Data,Class_1_Data,Class_2_Data] = Shuffle_data(bw_data);
    %% OutPut Value Prep
    Desired_out_temp = [class_2];
    Desired_out=repmat(Desired_out_temp,[1 4500]);
    %% Training 
    h = waitbar(0,'Training...');
    for iterator_i = 1:Iteration
        for iterator_j = 1:size(Shuffled_Data,2)
            Input_Layer_Weight = (Shuffled_Data{iterator_j}'*Weight_Input_Hidden);  %% Wx --> Input layer to first hidden layer
            Input_Layer_Weight_Bias = Input_Layer_Weight + Weight_Bias_Hidden';  %% Wx + b
            Hidden_Layer_Input = sigmoid(Input_Layer_Weight_Bias);   %% Sigmoid Activation Function
            Hidden_Layer_temp = (Hidden_Layer_Input *  Weight_Hidden) + Weight_Bias_Output';  %% Hidden Layer Inputs multiplied with Hidden Layer Weights
            Final_Output = sigmoid(Hidden_Layer_temp);
            %% Finding Error
            Err = Desired_out(iterator_j)'-Final_Output;   %% Difference b/w output and labelled output
            Delta = (Final_Output.*(1-Final_Output)).* Err;   %% Finding out the Partial derivative 
            %% Updading Weights
            Weight_Hidden=Weight_Hidden+(Learning_Rate*Hidden_Layer_Input'*Delta) + (Alpha.*Delta_Prev_Weight_Hidden);  %% Updating Hidden Layer Weights
            Weight_Bias_Output = Weight_Bias_Output + (2*Delta') + (Alpha*(Delta_Prev_Weight_Bias_Output));                 %% Updading Biases
            %% Updating Input Layer Weights
            Delta_Hidden = Hidden_Layer_Input'.*(1-Hidden_Layer_Input)'.*(Weight_Hidden*Delta');
            Weight_Input_Hidden=Weight_Input_Hidden+Learning_Rate*(Shuffled_Data{iterator_j}*Delta_Hidden') + (Alpha.*Delta_Prev_Weight_Input_Hidden);
            Weight_Bias_Hidden = Weight_Bias_Hidden + (2*Delta_Hidden) + (Alpha*Delta_Prev_Weight_Bias_Hidden);
            %% Delta For Momentum
            Delta_Prev_Weight_Hidden = Weight_Hidden - Prev_Weight_Hidden;
            Delta_Prev_Weight_Input_Hidden = Weight_Input_Hidden - Prev_Weight_Input_Hidden;
            Delta_Prev_Weight_Bias_Output  = Weight_Bias_Output - Prev_Weight_Bias_Output;
            Delta_Prev_Weight_Bias_Hidden  = Weight_Bias_Hidden - Prev_Weight_Bias_Hidden;
            Prev_Weight_Hidden = Weight_Hidden;
            Prev_Weight_Input_Hidden = Weight_Input_Hidden;
            Prev_Weight_Bias_Output = Weight_Bias_Output;
            Prev_Weight_Bias_Hidden = Weight_Bias_Hidden;
        end
            error(:,iterator_i)=Err;
            waitbar(iterator_i / Iteration)
    end
    close(h)
    %% Error Plots
    sqe=sum((error(:,1:iterator_i).^2),1);
    X = sprintf('Error is %f ',sqe(length(sqe)));
    disp(X);
    plot(sqe);
    title('Error Plot-Training');
    xlabel('Number Of Iterations');
    ylabel('Error^2');
    %% Saving Weights
    save('Weights','Shuffled_Data','Weight_Input_Hidden','Weight_Bias_Hidden','Weight_Hidden','Weight_Bias_Output',...
        'Class_1_Data','Class_2_Data');
end




