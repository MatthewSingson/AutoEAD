import pandas as pd
import numpy as np
import cv2 as cv
import dlib 

#rename to the filename of clip
cap = cv.VideoCapture('r02 - 02.1e.mp4')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

prevX = np.zeros(60)
prevY = np.zeros(60)
distance1 = []
i = 0

while True:
    ret, frame = cap.read()

    while (ret):
         
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv.circle(frame, (x, y), 4, (255, 0, 0), -1)

            #distances to measure - (52,58), (63,67), (50, 60), (54, 56), (52, 9), (51, 59), (53, 57)
            
            x52 = landmarks.part(51).x
            y52 = landmarks.part(51).y
            x58 = landmarks.part(57).x
            y58 = landmarks.part(57).y

            x63 = landmarks.part(62).x
            y63 = landmarks.part(62).y
            x67 = landmarks.part(66).x
            y67 = landmarks.part(66).y

            x50 = landmarks.part(49).x
            y50 = landmarks.part(49).y
            x60 = landmarks.part(59).x
            y60 = landmarks.part(59).y

            x54 = landmarks.part(53).x
            y54 = landmarks.part(53).y
            x56 = landmarks.part(55).x
            y56 = landmarks.part(55).y

            x51 = landmarks.part(50).x
            y51 = landmarks.part(50).y
            x59 = landmarks.part(58).x
            y59 = landmarks.part(58).y

            x53 = landmarks.part(52).x
            y53 = landmarks.part(52).y
            x57 = landmarks.part(56).x
            y57 = landmarks.part(56).y

            x09 = landmarks.part(8).x
            y09 = landmarks.part(8).y 

            #52,58
            xsq = (x52 - x58) ** 2
            ysq = (y52-y58) ** 2
            d5258 = round(np.sqrt(xsq + ysq), 3)

            #63, 67
            xsq = (x63 - x67) ** 2
            ysq = (y63 - y67) ** 2
            d6367 = round(np.sqrt(xsq + ysq), 3)

            #50,60
            xsq = (x50 - x60) ** 2
            ysq = (y50 - y60) ** 2
            d5060 = round(np.sqrt(xsq + ysq), 3)

            #54,56
            xsq = (x54 - x56) ** 2
            ysq = (y54 - y56) ** 2
            d5456 = round(np.sqrt(xsq + ysq), 3)

            #52,9
            xsq = (x52 - x09) ** 2
            ysq = (y52 - y09) ** 2
            d5209 = round(np.sqrt(xsq + ysq), 3)

            #51,59
            xsq = (x51 - x59) ** 2
            ysq = (y51 - y59) ** 2
            d5159 = round(np.sqrt(xsq + ysq), 3)

            #53,57
            xsq = (x53 - x57) ** 2
            ysq = (y53 - y57) ** 2
            d5357 = round(np.sqrt(xsq + ysq), 3)

            distance1.append(((i), d5258, d6367, d5060, d5456, d5209, d5159, d5357))
            i+=1
        
        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        #press escape to close
        if key == 27:
            break
        ret, frame = cap.read()
    break

df = pd.DataFrame(distance1, columns=['frame', '(52,58)', '(63,67)', '(50,60)', '(54,56)', '(52,09)', '(51,59)', '(53,57)'])
#rename to the filename of clip
df.to_csv('r07-03.csv',index=False)

cap.release()

