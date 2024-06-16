import cv2

#Function to generate the data set
def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)



#Function to draw the boundary/Rectangle around the face
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    colors = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0),"white":(255,255,255)}

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)  # Use colors[color] here
        textColor = colors['white']
        id=0
        id, _=clf.predict(gray_img[y:y+h, x:x+w])
        
        if id==1:
            cv2.putText(img, "Dheeraj", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, textColor, 1, cv2.LINE_AA)
        elif id==2:
            cv2.putText(img, "Tarun", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, textColor, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, textColor, 1, cv2.LINE_AA)
        

        coords = [x, y, w, h]

    return coords

#Function to recognize the face using the faces recorded in the data set
def recognize(img, clf, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0),"white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['white'], "Face",clf)
    return img





# #Function to detect the face and generate the data set
# def detect(img, faceCascade,img_id):
#     color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0),"white":(255,255,255)}
#     coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face",clf)
    
#     if(len(coords)==4):
#         #region of intrest(set only for face)
#         roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

#         user_id = 2
#         generate_dataset(roi_img, user_id, img_id)

#     return img




#load the face cascade classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

#initialize the video capture object
#for default webcam and if using phone type -1
video_capture = cv2.VideoCapture(0) 

# img_id = 0

while True:
    _, img = video_capture.read()
    cv2.imshow("Face Detection", img)   

   
 #using face detection function to detect and generate the data set
    # img = detect(img, faceCascade,img_id)



    #using face recognition function to recognize the face
    img=recognize(img,clf,faceCascade)

    cv2.imshow("Face Detection", img)

    # img_id += 1

    # Wait for a key press, or for the close button to be clicked
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or ESC key
        break

video_capture.release()
cv2.destroyAllWindows()