%% Author : Krishna Satish D S
function [Shuffled_Data,Class_1_Data,Class_2_Data] = Shuffle_data(datain)
    Shuffled_Data{100} = [];
    Class_1_Data{100} = [];
    Class_2_Data{100} = [];
    Class_1_Data = datain(1:550);
    Class_2_Data = datain(551:end);
    count = 1;
    for i=1:length(Class_1_Data)
        Shuffled_Data{count} = Class_1_Data{i};
        Shuffled_Data{count+1} = Class_2_Data{i};
        count = count+1;
    end
end
