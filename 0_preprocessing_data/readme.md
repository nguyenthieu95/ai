======================= Data Analysis ============================
- 2 type of data: stationary and non-stationary 
- stationary data --> model with more accuracy


A. Basic Feature Engineering With Time Series Data in Python
- Chung ta chi can nhung features co anh huong lon nhat trong viec hoc ra model relationship between 
the inputs (X) and the outputs (y) that we would like to predict.
- Goal of Feature Engineering: to provide strong and ideally simple relationships between new input
 features and the output feature for the supervised learning algorithm to model.
 
- Trong time series khong co input, output. Ta phai thiet ke ra input, output de ap dung vao supervised learning


1. Date Time Features: these are components of the time step itself for each observation.
- Ta co the trich xuat ra nhieu cau hoi de co the che' them du lieu
+ Minutes elapsed for the day.
+ Hour of day.
+ Business hours or not.
+ Weekend or not.
+ Season of the year.
+ Business quarter of the year.
+ Daylight savings or not.
+ Public holiday or not.
+ Leap year or not.



2. Lag Features: these are values at prior time steps.
- Phuong phap co dien nhat duoc su dung, don gian nhat.
- Dua vao thoi gian t-1 du doan thoi gian t+1 (sliding window = width = 1)

Shifted,    Original
NaN,        20.7
20.7,       17.9
17.9,       18.8

- Van de voi bai toan dang nay la width = ? thi ok --> Di thuc nghiem roi chon
- Hien tai chung ta moi chi su dung linear window. Co the nhung gia tri last week, last month, last year
se co ich hon. -> Se can nhung non-linear window



3. Window Features (Rolling Window Statistics): these are a summary of values over a fixed window of prior time steps.
- Thay vi add cac gia tri raw o thoi diem truoc do. Ta co the chi can add gia tri mean(cac gia tri truoc do)
- Ngoai mean ta co the su dung cac ham thong ke trong nay. Moi ham chuyen doi la 1 feature moi.
- Cach su dung mean thuong duoc dung nhat, va goi la: rolling mean

mean(t-2, t-1),     t+1
mean(20.7, 17.9),   18.8
19.3,               18.8

- Chung ta co the dung nhieu ham thong ke tren do nua vd nhu min, mean, max
#, Window Values
1, NaN
2, NaN, NaN
3, NaN, NaN, 20.7
4, NaN, 20.7, 17.9
5, 20.7, 17.9, 18.8

    min       mean   max   t+1
0   NaN        NaN   NaN  20.7
1   NaN        NaN   NaN  17.9
2   NaN        NaN   NaN  18.8
3   NaN        NaN   NaN  14.6
4  17.9  19.133333  20.7  15.8
5  14.6  17.100000  18.8  15.8
6  14.6  16.400000  18.8  15.8
7  14.6  15.400000  15.8  17.4
8  15.8  15.800000  15.8  21.8
9  15.8  16.333333  17.4  20.0


4. Expanding Window Statistics
- No cung kha giong voi rolling(). No liet ke tat ca nhung gia tri truoc do, sau do sum lai tao thanh feature moi.
#, Window Values
1, 20.7
2, 20.7, 17.9,
3, 20.7, 17.9, 18.8
4, 20.7, 17.9, 18.8, 14.6
5, 20.7, 17.9, 18.8, 14.6, 15.8








B. Transform non-stationary to stationary data
- De-trending is fundamental. This includes regressing against covariates other than time.
- Seasonal adjustment is a version of taking differences but could be construed as a separate technique.
- Transformation of the data implicitly converts a difference operator into something else; e.g., 
differences of the logarithms are actually ratios.
- Some EDA smoothing techniques (such as removing a moving median) could be construed 
as non-parametric ways of detrending. They were used as such by Tukey in his book on EDA.
 Tukey continued by detrending the residuals and iterating this process for as long as necessary 
 (until he achieved residuals that appeared stationary and symmetrically distributed around zero).
 
 1. Differencing is a popular and widely used data transform for time series.
- Dung de loai bo su phu thuoc vao time (temporal dependence) - no bao gom trends va seasonality
- Differencing can help stabilize the mean of the time series by removing changes in the level of 
a time series, and so eliminating (or reducing) trend and seasonality.

    difference(t) = observation(t) - observation(t-1)

a. Lag Difference
- Taking the difference between consecutive observations is called a lag-1 difference.
- For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality.

b. Difference Order
- Sau khi differencing thi temporal structure va~n co kha nang ton tai --> Ta se diffenrecing nhieu lan
- So lan diff do goi la difference order



2. 

















