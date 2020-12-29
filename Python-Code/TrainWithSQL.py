import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font
import sqlite3

window = tk.Tk()
window.title("Face_Recognation NQH")
window.configure(background='black')
window.geometry('1280x670')

lbl = tk.Label(window, text="Face Recognition Based Attendance System", bg="white" , fg="black" , width=50 , height=3, font=('times', 30, 'italic bold')) 
lbl.place(x=100, y=20)

lbl1 = tk.Label(window, text="Enter ID", width=20 , height=2 , fg="black" , bg="white", font=('times', 15, ' bold ') ) 
lbl1.place(x=200, y=200)

txt1 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt1.place(x=550, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold ')  )
txt2.place(x=550, y=315)

lbl3 = tk.Label(window, text="Notification →", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
lbl3.place(x=200, y=400)

message = tk.Label(window, text="", bg="white", fg="black", width=30, height=2, font=('times', 15, ' bold ')) 
message.place(x=550, y=400)
 
def clearId():
    txt1.delete(0, 'end')

def clearName():
    txt2.delete(0, 'end')

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
def InsertOrUpdate(Id,Name):
    conn=sqlite3.connect(".//Face_Recogniton.db")
  
    cmd='INSERT INTO People(ID,Name) values(?,?)'
    conn.execute(cmd,(Id,Name,))
    conn.commit()
    conn.close()

def takeImages():        
    Id=(txt1.get())
    name=(txt2.get())
    InsertOrUpdate(Id,name)
    if(isNumber(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        # harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("Python-Code/Image/ "+name +'.'+Id +'.'+ str(sampleNum) + '.jpg', gray[y:y+h,x:x+w])
                cv2.imshow('Face Detecting',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentRecord.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(isNumber(name)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(Id.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def trainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    # harcascadePath = "haarcascade_frontalface_default.xml"
    # detector =cv2.CascadeClassifier(harcascadePath)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces,Id = getImagesAndLabels("Python-Code/Image")
    recognizer.train(faces, np.array(Id))
    recognizer.save("Python-Code/DataSet/Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path1):
    imagePaths=[os.path.join(path1,f) for f in os.listdir(path1)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids
  
clearButton1 = tk.Button(window, text="Clear", command=clearId, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
clearButton1.place(x=850, y=200)

clearButton2 = tk.Button(window, text="Clear", command=clearName, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
clearButton2.place(x=850, y=300)  

takeImg = tk.Button(window, text="Take Images", command=takeImages, fg="black", bg="white", width=20, height=3, activebackground = "Green", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Train Images", command=trainImages, fg="black", bg="white", width=20, height=3, activebackground = "Green" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=800, y=500)

lbl4 = tk.Label(window, text="You, yes you are. Have a nice day.", width=80, fg="white", bg="black", font=('times', 15, ' bold')) 
lbl4.place(x=200, y=620)

window.mainloop()