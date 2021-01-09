%% Author : Krishna Satish D S
function dataout = Normalization(datain)
    [min, max] = Min_Max_Finder(datain);
    dataout = datain - min;
    dataout = (dataout/range(dataout(:)))*(max-min);
    dataout = dataout + min;
end
