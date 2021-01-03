https://www.rs-online.com/designspark/google-teachable-machine-raspberry-pi-4-cn

選擇7 Advanced Options" à "A1  Expand filesystem出現錯誤：
your partition layout is not currently supported by this
tool. you are probably using NOOBS
解決：不管，跳過

Step5.  安裝OpenCV套件======
上面網址安裝不全，參考以下網址：
https://qengineering.eu/install-opencv-4.1-on-raspberry-pi-4.html

sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install libjasper-dev libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install liblapack-dev gfortran
sudo apt-get install gcc-arm*
sudo apt-get install protobuf-compiler

ln -s /usr/local/python/cv2/python-3.7/cv2.cpython-37m-arm-linux-gnueabihf.so cv2.so

樹莓派要使用USB Webcam分類器：
	在執行程式之前需要下載分類器的檔案，連結: https://reurl.cc/4R3dVL 
	下載TM2_tflite.py，放在converted_tflite_quantized裡
若要使用樹莓派專用鏡頭，請下載官方提供的 label_image.py(✖未測試)：
	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/

開啟RPi的終端機執行：
python3 TM2_tflite.py --model model.tflite --labels labels.txt

==================================
google teachable machine訓練專案儲存在：
	https://teachablemachine.withgoogle.com/train
	點選「open project from Drive」選「My Imaage Model.tm」
	


