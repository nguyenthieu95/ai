- Bắt đầu viết mô hình với các bộ tham số chạy song song trên server.

1. Tạo file gen các bộ trọng số lưu vào dạng csv hoặc json 
2. Tách riêng các hàm helper ra 
3. Tách riêng các hàm preprocessing data ra 
4. Tách riêng hàm train ra . Hàm trên nhận vào bộ tham số và trả lại model 
5. Tách riêng hàm predict dựa vào model 
6. Tách riêng các hàm vẽ, lưu csv, png 

7. Viết code sử dụng : joblib chạy song song 
8. Sử dụng subprocess của python để chạy command line 

9. Log kết quả ra file csv hoặc dưới dạng json rồi lưu lại vào database 
10. Database có thể có dạng như: 
    experiment_id, model_name, params( dạng json), score
    - sau rồi order by score -> Sẽ ra kết quả giữa các model 
    - score ở đây tính cả MAE, RMSE, MAPE 
     
- Giờ các bộ tham số không phải cho vào vòng for nữa.
- Ta đọc các bộ tham số từ file csv hoặc json rồi đưa vào queue. 
- Sau đó mở các tiến trình song song để chạy model train với từng bộ tham số đó.
- Khi mà đọc 1 bộ tham số từ queue ra thì bộ tham số sẽ bị xóa đi trong queue đó.
- Khi mất điện, tiến trình chết, các bộ tham số còn lại chưa được chạy vẫn ở trong queue. 


====== Script 1 =========

SONIA:

cpu: 31154              run
ram: 31445              run
multi_cpu: 31528        run
multi_ram: 31876        run


SoBee:

cpu: 31999              run
ram: 32356              run
multi_cpu: 470          run
multi_ram: 1173         run


============ Script 2 ===============

SONIA:

cpu: 9223           run
ram: 9260           run
multi_cpu: 9303     run
multi_ram: 9454     run


SoBee:

cpu:        done
ram:        done
multi_cpu:  done
multi_ram:  done

============ Script 3 ===============
Test1:

cpu:  done
ram:  done


Test2:

cpu:  done 
ram:  done 


Test3:

cpu:  done 
ram:  done 





