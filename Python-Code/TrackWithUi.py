import tkinter as tk
from tkinter import Message, Text
import cv2,os
import csv
import pandas as pd
import datetime
import time
import tkinter.font as font
import pyrebase
from firebase import firebase
from google.cloud import storage
from google.cloud.storage.blob import Blob

window = tk.Tk()
window.title("Face_Recognition ")
window.configure(background='white')
window.geometry('1280x670')

lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white", fg="black", width=50, height=3, font=('times', 30, 'italic bold')) 
lbl.place(x=100, y=20)

lbl1 = tk.Label(window, text="↓  List Of Present Students  ↓", width=25, fg="black", bg="white", height=2, font=('times', 15, ' bold')) 
lbl1.place(x=540, y=320)

message = tk.Label(window, text="", fg="black", bg="white", activeforeground = "green", width=35, height=7, font=('times', 15, ' bold ')) 
message.place(x=470, y=400)

config = {
  "apiKey": "AIzaSyDaQcVMyMgkQmT7EoTcPow6bOo-B9eaSWw",
  "authDomain": "facereconition-431f9.firebaseapp.com",
  "projectId": "facereconition-431f9",
  "databaseURL": "https://facereconition-431f9-default-rtdb.firebaseio.com",
  "storageBucket": "facereconition-431f9.appspot.com",
  "messagingSenderId": "642588267941",
  "appId": "1:642588267941:web:0493c40b9d576d2bee844c",
  "measurementId": "G-82250C3WQS"
}

firebase = firebase.FirebaseApplication("https://facereconition-431f9-default-rtdb.firebaseio.com/", None) #databaseURL
blob = Blob.from_string("gs://facereconition-431f9.appspot.com") #storageBucket

def trackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Python-Code/DataSet/Trainner.yml")
    # harcascadePath = "haarcascade_frontalface_default.xml"
    # faceCascade = cv2.CascadeClassifier(harcascadePath)   
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    df=pd.read_csv("StudentRecord.csv")
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.3,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                                              
            if(conf<50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                # cv2.putText(im, "Name: " + str(df.loc['Name'].values), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("Python-Code/UnknownImages"))+1
                cv2.imwrite("Python-Code/UnknownImages/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first') 

        cv2.imshow('Face Recognizing',im)
        

        if cv2.waitKey(10000)==ord('q'):
            break

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")

    #fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    fileName="Python-Code/Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+".csv"
    attendance.to_csv(fileName,index=False)
    Firebase = pyrebase.initialize_app(config)
    storage = Firebase.storage()
    blob = storage.child('uploads1/'+ fileName).put(fileName)
    
    data =  { 'name': "Date_"+date+"  Time_"+Hour+"-"+Minute+"-"+Second, 'url': "https://console.firebase.google.com/u/6/project/facereconition-431f9/storage/facereconition-431f9.appspot.com/files~2Fuploads1~2FPython-Code~2FAttendance%5CAttendance_"+date+"_"+Hour+"-"+Minute+".csv?alt=media&token="+blob['downloadTokens']}
    #data =  { 'name': "Date_"+date+"  Time_"+Hour+"-"+Minute+"-"+Second, 'url': ULR stograge to uploade file excel(.csv)  }
    result = firebase.post('facereconition-431f9-default-rtdb/uploads1/',data) #databaseURL + folder you create
    print(result)

    cam.release()
    cv2.destroyAllWindows()

    res=attendance
    message.configure(text= res)

trackImg = tk.Button(window, text="Track Image", command=trackImages, fg="black", bg="white", width=20, height=3, activebackground = "Yellow", font=('times', 15, ' bold '))
trackImg.place(x=400, y=200)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red", font=('times', 15, ' bold '))
quitWindow.place(x=700, y=200)

lbl3 = tk.Label(window, text="FaceRecognition", width=80, fg="white", bg="black", font=('times', 15, ' bold')) 
lbl3.place(x=200, y=620)

window.mainloop()