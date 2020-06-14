# People Semantic Segmentation Pytorch
## 1, Giới thiệu
-   Mô hình segmenation: Deeplab v3 plus
-   Backbone mode: Resnet 50
## 2, Dữ liệu
- Tài bộ dữ liệu **Supervisely Person** [**tại đây**](https://drive.google.com/file/d/1elfk_otCj79g4zPR7e_JCov0fjx_Xs8w/view?usp=sharing) 
- Chạy câu lệnh sau để tiến hành xử lí dữ liệu: 
## 3, Cài đặt 
- Cài đặt **virtualenv** để cài đặt môi trường ảo cho project 
- Cài đặt các thư viện cần thiết từ file **requirement.txt**: ```pip3 install -r requirement.txt```
- Để huấn luyện mô hình: ```python3 train.py```
- Tại model [**tại đây**](https://drive.google.com/file/d/1QQAcT8CJ4-65xDw6KDdIRZgdnGdbx7tp/view?usp=sharing), để model tại thư mục **model/**
- Chạy model tại localhost: ```python3 app.py```
- Mở ứng dụng postman và tạo 1 request tới server: 
	+ ```localhost:5000/people-segmentation``` với phương thức **POST**
	+ Ở mục **body** chọn **form-data** với:
		* key: file, chọn File
		* value: Ảnh cần segment (đảm bảo có người, không có người thì ảnh đen thui)
	+ Server chưa kịp code chức năng trả ảnh cho người dùng, chỉ mới show ảnh trên server để kiểm tra
