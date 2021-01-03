from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
#import picamera
import cv2

import RPi.GPIO as GPIO   #GPIO套件
#定義GPIO接腳 
pinLED1 = 26    #沒口罩
pinLED2 = 19    #有口罩
pinLED3 = 13    #空


GPIO.setmode(GPIO.BCM) #定義GPIO接腳編號方式
GPIO.setwarnings(False)
GPIO.setup(pinLED1, GPIO.OUT)  #定義GPIO接腳模式
GPIO.setup(pinLED2, GPIO.OUT)
GPIO.setup(pinLED3, GPIO.OUT)

#import tensorflow as tf

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  #interpreter = tf.lite.Interpreter(args.model)
  interpreter = Interpreter(args.model)

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  #with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    #camera.start_preview()

  cap = cv2.VideoCapture(0)
  #擷取畫面 寬度 設定為640
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
  #擷取畫面 高度 設定為480
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  key_detect = 0
  times=1
  while (key_detect==0):
    ret,image_src =cap.read()

    frame_width=image_src.shape[1]
    frame_height=image_src.shape[0]

    cut_d=int((frame_width-frame_height)/2)
    crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

    image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

    start_time = time.time()
    if (times==1):
      results = classify_image(interpreter, image)
      elapsed_ms = (time.time() - start_time) * 1000
      label_id, prob = results[0]

      print(labels[label_id],prob)
    cv2.putText(crop_img,labels[label_id] + " " + str(round(prob,3)), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

    # 簡單的IF判斷程序控制GPIO輸出
    if labels[label_id] == "0 nomask":  
        GPIO.output(pinLED1, 0)
        GPIO.output(pinLED2, 0)
        GPIO.output(pinLED3, 0)
        GPIO.output(pinLED1, 1) 
        time.sleep(1)
        
    elif labels[label_id] == "1 mask":
        GPIO.output(pinLED1, 0)
        GPIO.output(pinLED2, 0)
        GPIO.output(pinLED3, 0)
        GPIO.output(pinLED2, 1)
        time.sleep(1)
        
    elif labels[label_id] == "2 nothing":
        GPIO.output(pinLED1, 0)
        GPIO.output(pinLED2, 0)
        GPIO.output(pinLED3, 0)
        GPIO.output(pinLED3, 1) 
        time.sleep(1)

    times=times+1
    if (times>1):
      times=1

    cv2.imshow('Detecting....',crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
