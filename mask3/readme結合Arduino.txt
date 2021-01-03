口罩辨識訓練機 Google Teachable Machine Raspberry Pi結合Arduino
★TM2_tflite.py有修改

https://bit.ly/2MtgeDP

執行程式儲存在RPi的/home/pi/mask3

開啟RPi的終端機執行：
python3 TM2_tflite.py --model model.tflite --labels labels.txt

目前較佳(pi還是跑不會)--
訓練張數：1000
訓練次數：50
圖pixel：16

==================================
google teachable machine訓練專案儲存在：
	https://teachablemachine.withgoogle.com/train
	點選「open project from Drive」選「mask3.tm」
	
Arduino專案：
	ArduinoSample/BlocklyDuino/AI/Mask3