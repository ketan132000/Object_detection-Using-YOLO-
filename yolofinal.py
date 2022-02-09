import cv2
import time
import numpy as np
#%%
classes=[]
with open('coco.names', 'r') as f:
    obj_names= f.read().splitlines()
classes = [name.split(',')[0] for name in obj_names]
#print(classes)
#%%
net= cv2.dnn.readNet(model='yolov3.weights',config='yolov3.cfg',framework="DNN")
cap=cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('video_result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))
#%%
while cap.isOpened():
    ret, frame = cap.read()
    img=frame
    if ret:
        height, width, _ = img.shape
        blob=cv2.dnn.blobFromImage(img,1/255,(220,220),(0,0,0),swapRB=True,crop=False)
        net.setInput(blob)
        start = time.time()
        output_layers_names= net.getUnconnectedOutLayersNames()
        layerOutputs=net.forward(output_layers_names)
        end = time.time()
        fps = 1 / (end-start)
        print("FRAMES PER SECOND:",fps,"FPS")
        boxes=[]
        confidences=[]
        class_ids=[]
    
        for output in layerOutputs:
            for detection in output:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence = scores[class_id]
                if confidence>0.7:
                    centre_x=int(detection[0]*width)
                    centre_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)
                
                    x=int(centre_x-w/2)
                    y=int(centre_y-h/2)
                
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
       # print(len(boxes))      
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        font= cv2.FONT_HERSHEY_PLAIN
        colors= np.random.uniform(0,255,size=(len(boxes),3))
   
        if len(indexes)> 0:
        
            for i in indexes.flatten():
                x,y,w,h= boxes[i]
                label= str(classes[class_ids[i]])
                confidence=str(round(confidences[i],2))
                color=colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label+" "+confidence,(x,y-10),font,1,(255,255,255),2)
                #cv2.putText(img,"fps=",fps, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255) ,2)
        cv2.imshow("image",img)
        #out.write(img)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
    else:
        break
cap.release()  
cv2.destroyAllWindows()
