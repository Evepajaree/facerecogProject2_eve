import cv2
import time
import requests



url = 'https://notify-api.line.me/api/notify'
token = '7qnFod01ilxe5CuuRbgfvZxz8s4aic52InDduUSmeWv'
headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}


def line(String):
		msg = {String}
		r = requests.post(url, headers=headers, data = {'message':msg})
            
		



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def create_dataset(img,id,img_id):
      cv2.imwrite("dataset/pic."+str(id)+"."+str(img_id)+".jpg",img)
      

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf): 
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
      coords=[]
      for(x,y,w,h) in features:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            id,con = clf.predict(gray[y:y+h,x:x+w])
            
            
            if id ==1 :
                  
                  if 50>=con<=100:
                        text = "Eve"
                        cv2.putText(img,"Eve",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                        line(text)
                        

                  else :
                        text = "Unknow"
                        cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                        line(text)
                        
         
      return img,coords

      
def detect(img,faceCascade,img_id,clf): #clf
      img,coords= draw_boundary(img,faceCascade,1.1,10,(255,0,0),clf)
      id=1
      if len(coords)==4:
            id =1
            result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            
      return img
      


def show(img):
      cv2.imshow('frame',img)

img_id=0

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.1.35:21702/videostream.cgi?user=admin&pwd=pajaree1418')

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while(True):
      ret,frame = cap.read()
      if ret == True:
            frame = detect(frame,faceCascade,img_id,clf)
            cv2.imshow('frame',frame)
            img_id+=1
      
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                  break


cap.release()
#cv2.destoyAllWindows()
cv2.waitKey(0)

