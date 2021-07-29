import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_dataset(img,id,img_id):
      cv2.imwrite("dataset/pic."+str(id)+"."+str(img_id)+".jpg",img)
      

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text): #clf
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors) #set classifier
      coords=[]
      for(x,y,w,h) in features:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            coords=[x,y,w,h]
      return img,coords

def detect(img,faceCascade,img_id): #clf
      img,coords= draw_boundary(img,faceCascade,1.1,10,(255,0,0),"Face")
      id=2
      if len(coords)==4:
            id = 2
            result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            #y+h
            create_dataset(result,id,img_id)
      return img

img_id= 2
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.1.35:21702/videostream.cgi?user=admin&pwd=pajaree1418')


while(True):
      r,frame = cap.read()
      frame=detect(frame,faceCascade,img_id) 
      cv2.imshow('frame',frame)
      
      img_id+=1
      if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
cap.release()
cv2.destoyAllWindows()

