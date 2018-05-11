Screen server Id

sonia:

    3m:
        cpu         :
        ram         :
        multi_cpu:  
        multi_ram:  
        
    5m:
        cpu         :
        ram         :
        multi_cpu:      
        multi_ram:     
        
    8m:
        cpu         :
        ram         :
        multi_cpu:  
        multi_ram:  
        
    10m:
        cpu         :
        ram         :
        multi_cpu:  
        multi_ram: 


sopso

    3m:
        cpu         :3911
        ram         :4023
        multi_cpu:  4087
        multi_ram:  4156
        
    5m:
        cpu         : 4226
        ram         : 4292
        multi_cpu:  4357
        multi_ram:  4423
        
    8m:
        cpu         :4491
        ram         :4555
        multi_cpu:  4619
        multi_ram:  4687
        
    10m:
        cpu         :4756
        ram         :4821
        multi_cpu:  4885
        multi_ram:  4952
        
        

sobee

    3m:
        cpu         :6752
        ram         :6816
        multi_cpu:  6880
        multi_ram:  6947
        
    5m:
        cpu         :7019
        ram         :7085
        multi_cpu:  7150
        multi_ram:  7216
        
    8m:
        cpu         :
        ram         :
        multi_cpu:  
        multi_ram:  
        
    10m:
        cpu         :
        ram         :
        multi_cpu:  
        multi_ram:  
        
        
        
Ann (con may ao tang 9):
    
    3m:
        cpu         : 19532
        ram         : 13545
        multi_cpu:   13593
        multi_ram:  13640
        
    5m:
        cpu         : 13690
        ram         : 13736
        multi_cpu:   13781
        multi_ram:   13830
        
    8m:
        cpu         : 13879
        ram         : 13908
        multi_cpu:   13969
        multi_ram:   14000
        
    10m:
        cpu         : 14065
        ram         : 14094
        multi_cpu:   14139
        multi_ram:  14187


        
Kq cua 1 screen: 3^7 = 2187 (model)
    ->16 screen: 34992     

Các file data_resource_usage_3Minutes_6176858948.csv
	data_resource_usage_5Minutes_6176858948.csv
	data_resource_usage_8Minutes_6176858948.csv
	data_resource_usage_10Minutes_6176858948.csv
lần lượt là time series tại các điểm thời gian cách nhau 3,5,8,10 phút
của jobid 6176858948. Job id này có 25954362 bản ghi dữ liệu chạy trong khoảng thời gian 29 ngày.
Thứ tự các cột lần lượt là:
time_stamp,numberOfTaskIndex,numberOfMachineId,meanCPUUsage,canonical memory usage,AssignMem,unmapped_cache_usage,page_cache_usage,max_mem_usage,mean_diskIO_time,
mean_local_disk_space,max_cpu_usage, max_disk_io_time, cpi, mai,sampling_portion,agg_type,sampled_cpu_usage
Kết quả dự đoán với LSTM sử dụng keras. 
Các cột sử dụng để dự đoán meanCPUUsage, canonical memory usage.
kết quả

CPU 		sliding =2	sliding = 3	sliding = 4	sliding = 5
Multivariate	0.3221		0.3318		0.3383		0.3259		
Univariate	0.3510		0.3316		0.3528		0.3278

Memory 		sliding =2	sliding = 3	sliding = 4	sliding = 5
Multivariate	0.0303		0.0305		0.0309		0.0307	
Univariate	0.0357		0.0346		0.0406		0.0362


3m -> 13920 data 
5m -> 8352 data 
8m -> 5220 data 
10m -> 4176 data 

- Nếu có validation lấy tỉ lệ: 70-20-10 (train-validation-test) 
- Nếu không có validation thì lấy tỉ lệ: 80-20  (train-test) 

- Validation :
    + 3m (13900): 9730-2780-1390
    + 5m (8300) : 5810-1660-830
    + 8m (5200) : 3640-1040-520
    + 10m (4100): 2870-820-410

- No-Validation:
    + 3m (13900): 11120-2780
    + 5m (8300) : 6640-1660
    + 8m (5200) : 4160-1040
    + 10m (4100): 3280-820

- univariate thì chạy:
    cpu -> cpu 
    ram -> ram
- multivariate thì chạy:
    cpu, ram --> cpu
    cpu, ram --> ram 
Nếu có thể thì chạy thêm cả: cpu, ram --> cpu, ram 

- sliding window thì dùng: 2, 3, 4, 5 
Vd: Nếu sliding = 3

    X                       y
    t-3, t-2, t-1           t 
    
    X                       y
    medium(t-3, t-2, t-1)   t 
    
    X                                                                   y 
    min(t-3, t-2, t-1)   median(t-3, t-2, t-1)   max(t-3, t-2, t-1)     t 

Nếu là multivariate thì dùng:

    X               Z                    y                                   y
    t-3, t-2, t-1, z-3, z-2, z-1        t       hoặc nếu 2 đầu ra thì :     t, z
hoặc 
    X                             Z                     y
    medium(t-3, t-2, t-1)  medium(z-3, z-2, z-1)        t hoăc t, z
hoặc 
    X                                                       y
    min(t-3,t-2,t-1) median(t-3,t-2,t-1) max(t-3,t-2,t-1)  min(z-3,z-2,z-1) median(z-3,z-2,z-1) max(z-3,z-2,z-1)   t hoặc t,z 
    
    
    
        
    
