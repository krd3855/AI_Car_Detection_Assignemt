            Input_Layer_Weight = normalize(bw_data'*Weight_Input_Hidden);  %% Wx --> Input layer to first hidden layer
            Input_Layer_Weight_Bias = Input_Layer_Weight + Weight_Bias_Hidden';  %% Wx + b
            Hidden_Layer_Input = sigmoid(Input_Layer_Weight_Bias);   %% Sigmoid Activation Function
            Hidden_Layer_temp = (Hidden_Layer_Input *  Weight_Hidden) + Weight_Bias_Output';  %% Hidden Layer Inputs multiplied with Hidden Layer Weights
            Final_Output = sigmoid(Hidden_Layer_temp)