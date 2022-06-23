##################
# Import Section #
##################

# General Imports
from __future__ import division
# from time import sleep
import time
import shutil

# Imports needed for DC Motor Control

# Imports needed for Servo Control

# Imports needed for Camera and AI
from tflite_runtime.interpreter import Interpreter 
from PIL import Image
from picamera import PiCamera
import numpy as np


####################
# Function Section #
####################
  
#functions for the classification model
def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]

########################
# Main Program Section #
########################

print("\n")
print("The Brick Sorter starts.....")
print("\n")

ExecutionTime = 10

#start the model
data_folder = "/home/pi/Desktop/Lego_Maschine/TFLite_MobileNet/"
model_path = data_folder + "TF_Lite_600_precise_model.tflite"
label_path = data_folder + "TF_Lite_600_labels.csv"
interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
# print("Image Shape (", width, ",", height, ")")

#start both belts (motor 4 and 5)

#start the camera
camera = PiCamera()
camera.rotation = 180
camera.resolution =(224, 224)
camera.brightness =70
camera.contrast =70
camera.awb_mode ='auto'
camera.exposure_mode ='auto'
camera.start_preview(alpha=255)

#start the processing loops (ideally one brick gets processed each loop)
dummyCounter = 0
while (dummyCounter < ExecutionTime):
   
    BrickFound = False

    while BrickFound == False:
    
        time1 = time.time()
        camera.capture('/home/pi/Pictures/Pic_Lego_Brick/myBrickImage.jpg')
        time2 = time.time()
        # Load an image to be classified.
        image = Image.open("/home/pi/Pictures/Pic_Lego_Brick/myBrickImage.jpg").convert('RGB').resize((width, height))
        # Classify the image.
        label_id, prob = classify_image(interpreter, image)
        # Read class labels.
        labels = load_labels(label_path)
        # Return the classification label of the image.
        classification_label = labels[label_id]
        time3 = time.time()
        snapshot_time = np.round(time2-time1, 3)               
        classification_time = np.round(time3-time2, 3)               
        if not classification_label == "NoBrick" and prob>0.5:
            print("Loop:",dummyCounter, " - Label ID:", label_id, " - Label Name:", classification_label, " - Accuracy:", np.round(prob*100, 2), "% - Classification Time:", classification_time, " - Snapshot Time:" ,snapshot_time)
            BrickFound = True
            VictoryLabel_ID = label_id
            shutil.copy("/home/pi/Pictures/Pic_Lego_Brick/myBrickImage.jpg","/home/pi/Pictures/Pic_Lego_Brick/myVictoryBrickImage_loop" + str(dummyCounter) + ".jpg")
                    
    if VictoryLabel_ID == 0:
        go_to_bucket_21()
    if VictoryLabel_ID == 1:
        go_to_bucket_22()
    if VictoryLabel_ID == 2:
        go_to_bucket_23()
    if VictoryLabel_ID == 3:
        go_to_bucket_24()
    if VictoryLabel_ID == 4:
        go_to_bucket_9()
    if VictoryLabel_ID == 5:
        go_to_bucket_20()
    if VictoryLabel_ID == 6:
        print("this is the empty picture label !!")
    if VictoryLabel_ID == 7:
        go_to_bucket_23()
        
    dummyCounter = dummyCounter + 1

#end section
camera.stop_preview()
print("\n")
print("The Brick Sorter ends.....")
print("\n")


