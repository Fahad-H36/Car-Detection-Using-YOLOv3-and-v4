
# !wget -P "/content/drive/MyDrive/yolo v3" "https://pjreddie.com/media/files/yolov3.weights"

# !git -C "/content/drive/MyDrive/yolo v3" clone https://github.com/pjreddie/darknet

# !pip install numpy

# !wget "https://www.pexels.com/video/5382495/download/?search_query=dash%20cam&tracking_id=0pjbeap2vdd" -p "/content/drive/MyDrive/yolo v3"

import numpy as np
import cv2

def detect(video):

  net = cv2.dnn.readNet("yolov4_tiny.weights", "yolov4_tiny.cfg")
    
  with open("coco.names", 'r') as f:
    classes = list(f.read().splitlines())


  cap = cv2.VideoCapture(video)  
  
  # img = cv2.imread("/content/drive/MyDrive/yolo v3/dashcams-2048px-0096.jpg")
  while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416,416), mean=(0,0,0), swapRB=False, crop=False)
    
    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    layer_Output = net.forward(output_layers) 

    class_ids = []
    confidences = []
    boxes = []

    for output in layer_Output:
      for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence>0.5:
          center_x = int(detection[0]*width)
          center_y = int(detection[1]*height)
          w = int(detection[2]*width)
          h = int(detection[3]*height)

          x = int(center_x - w/2)
          y = int(center_y - h/2)

          boxes.append([x, y, w, h])
          confidences.append(float(confidence))
          class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes),3))
    # img = cv2.imread("/content/drive/MyDrive/yolo v3/dashcams-2048px-0096.jpg")

    if len(indexes)>0:


      for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x,y+15), (x+w, y+h+15), color, 4)
        cv2.putText(img, label+" "+confidence, (x,y+20), font, 2, color, 3)
      
      img = cv2. resize(img, (960, 540))
      cv2.imshow("img", img)
      key = cv2.waitKey(1)
      if key == 27:
        break

  cap.release()
  cv2.destroyAllWindows()

detect("Video Path")




