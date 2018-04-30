output = list()
for i in range(1, 2) :
    arr = "1998-06-30 19:53:00,137451.0".split(',')
    print arr
    output.append(float(arr[1]))
    if arr[1] == '':
        arr[1] = 0