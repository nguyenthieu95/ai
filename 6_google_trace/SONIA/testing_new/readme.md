How to features engineering
    https://www.quora.com/How-can-I-extract-the-features-from-time-series
    
    Mean - The DC component (average value) of the signal over the window
    Median - The median signal value over the window
    Standard Deviation - Measure of the spreadness of the signal over the window
    Variance - The square of standard deviation
    Root Mean Square - The quadratic mean value of the signal over the window
    Averaged derivatives - The mean value of the first order derivatives of the signal over the window
    Skewness - The degree of asymmetry of the sensor signal distribution
    Kurtosis - The degree of peakedness of the sensor signal distribution
    Interquartile Range - Measure of the statistical dispersion, being equal to the difference between the 75th and the 25th percentiles of the signal over the window
    Zero Crossing Rate - The total number of times the signal changes from positive to negative or back or vice versa normalized by the window length
    Mean Crossing Rate - The total number of times the signal changes from below average to above average or vice versa normalized by the window length
    Pairwise Correlation - Correlation between two axes (channels) of each sensor and different sensors
    Spectral Entropy - Measure of the distribution of frequency components

Feature extraction from time series
    https://edux.fit.cvut.cz/oppa/MI-PDD/prednasky/l8-signal-extraction.pdf
    
    https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
    

- Mutation: code dùng cluster + mutation + backpropagation 
- No_mutation: cluster + backpropagation 
- cluster_flnn: cluster + mạng FLNN 
- cluster_ga: cluster + mutation/ no mutation + genetic algorithm(GA)
- cluster_pso: cluster + mutation / no mutation + partical social optimization 


cluster_pso/mutation/2m : Chua code dau tien o day (2 loai PSO duoc code ra tu day)
- loai 1: Update gbest sau moi~ lan co 1 con chim di chuyen
- Loai 2: Update gbest sau moi~ lan ca da`n chim di chuyen


Update tensorflow: 
    pip list | grep tensorflow
    sudo pip install --upgrade pip
    sudo pip install --upgrade tensorflow
    sudo pip install --upgrade tensorflow-gpu
    


Chay cluster_pso

mutation
    2m
        cpu,ram: 14341 die
        cpu:     14420  
        ram:     14507 die
    5m
        cpu,ram: 14568 die
        cpu:     14630
        ram:     14696 die
    10m
        cpu,ram: 14759 die
        cpu:     14821
        ram:     14887 die 

no_mutation
    2m
        cpu,ram: 14955  die
        cpu:     15017
        ram:     15173  die
        2para - 1out | cpu, ram : 21232
    5m
        cpu,ram: 15302    die
        cpu:     15364  
        ram:     19593    die
        2para - 1out | cpu, ram : 21299
    10m
        cpu,ram: 15583  die 
        cpu:     15644 
        ram:     15706 die
        2para - 1out | cpu, ram : 21367
        
        
        


        
        
        
        
    
    
