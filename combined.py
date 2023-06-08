import cv2
import mediapipe as mp
import dlib
import numpy as np
import pandas as pd 

#rename to the filename of clip
cap = cv2.VideoCapture("r07-01.mp4")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

xArr = []
yArr = []
distance1 = []
i = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence  = 0.5, min_tracking_confidence = 0.5) as holistic:
    while True:
        ret, image = cap.read()
        if ret is not True:
            break
        height, width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Facial landmarks
        resultHands = hands.process(rgb_image)
        resultHolistic = holistic.process(image)

        height, width, _ = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
                xArr.append(x)
                yArr.append(y)
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

            #distances to measure - (52,58), (63,67), (50, 60), (54, 56), (52, 9), (51, 59), (53, 57), (49, 7), 
            # (55,11), (57,10), (56,11), (52, 14), (52, 4), (63, 9), (60, 6), (56,12), (53,10), (51, 8)

            #52,58
            xsq = (xArr[51] - xArr[57]) ** 2
            ysq = (yArr[51]-yArr[57]) ** 2
            d1 = round(np.sqrt(xsq + ysq), 3)

            #63, 67
            xsq = (xArr[62] - xArr[66]) ** 2
            ysq = (yArr[62]-yArr[66]) ** 2
            d2 = round(np.sqrt(xsq + ysq), 3)

            #50,60
            xsq = (xArr[49] - xArr[59]) ** 2
            ysq = (yArr[49]-yArr[59]) ** 2
            d3 = round(np.sqrt(xsq + ysq), 3)

            #54,56
            xsq = (xArr[53] - xArr[55]) ** 2
            ysq = (yArr[53]-yArr[55]) ** 2
            d4 = round(np.sqrt(xsq + ysq), 3)

            #52,9
            xsq = (xArr[51] - xArr[8]) ** 2
            ysq = (yArr[51]-yArr[8]) ** 2
            d5 = round(np.sqrt(xsq + ysq), 3)

            # #51,59
            xsq = (xArr[51] - xArr[58]) ** 2
            ysq = (yArr[51]-yArr[58]) ** 2
            d6 = round(np.sqrt(xsq + ysq), 3)

            #53,57
            xsq = (xArr[52] - xArr[56]) ** 2
            ysq = (yArr[52]-yArr[56]) ** 2
            d7 = round(np.sqrt(xsq + ysq), 3)

            #49,7
            xsq = (xArr[48] - xArr[6]) ** 2
            ysq = (yArr[48]-yArr[6]) ** 2
            d8 = round(np.sqrt(xsq + ysq), 3)

            #55,11
            xsq = (xArr[54] - xArr[10]) ** 2
            ysq = (yArr[54]-yArr[10]) ** 2
            d9 = round(np.sqrt(xsq + ysq), 3)

            #57,10
            xsq = (xArr[56] - xArr[9]) ** 2
            ysq = (yArr[56]-yArr[9]) ** 2
            d10 = round(np.sqrt(xsq + ysq), 3)

            #56,11
            xsq = (xArr[55] - xArr[10]) ** 2
            ysq = (yArr[55]-yArr[10]) ** 2
            d11 = round(np.sqrt(xsq + ysq), 3)

            #52,14
            xsq = (xArr[51] - xArr[13]) ** 2
            ysq = (yArr[51]-yArr[13]) ** 2
            d12 = round(np.sqrt(xsq + ysq), 3)

            #52,4
            xsq = (xArr[51] - xArr[3]) ** 2
            ysq = (yArr[51]-yArr[3]) ** 2
            d13 = round(np.sqrt(xsq + ysq), 3)

            #63,9
            xsq = (xArr[62] - xArr[8]) ** 2
            ysq = (yArr[62]-yArr[8]) ** 2
            d14 = round(np.sqrt(xsq + ysq), 3)

            #60,6
            xsq = (xArr[59] - xArr[5]) ** 2
            ysq = (yArr[59]-yArr[5]) ** 2
            d15 = round(np.sqrt(xsq + ysq), 3)

            #56,12
            xsq = (xArr[56] - xArr[11]) ** 2
            ysq = (yArr[56]-yArr[11]) ** 2
            d16 = round(np.sqrt(xsq + ysq), 3)

            #53,10
            xsq = (xArr[52] - xArr[9]) ** 2
            ysq = (yArr[52]-yArr[9]) ** 2
            d17 = round(np.sqrt(xsq + ysq), 3)

            #51,8
            xsq = (xArr[50] - xArr[7]) ** 2
            ysq = (yArr[50]-yArr[7]) ** 2
            d18 = round(np.sqrt(xsq + ysq), 3)

            distance1.append((i, d1, d2, d3,d4, d5,d6,d7,d8,d9,d10, d11, d12,d13,d14,d15,d16,d17,d18))
            #distance1.append(((i), d5258, d6367, d5060, d5456, d5209, d5159, d5357))
            print(xArr)
            print(yArr)
            i+=1
            xArr = []
            yArr = []
        
        if resultHands.multi_hand_landmarks:
            for handLms in resultHands.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                
        
        mpDraw.draw_landmarks(image, resultHolistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        #press escape to close
        if key == 27:
            break
    
df = pd.DataFrame(distance1, columns=['frame', 'distance1', 'distance2', 'distance3', 'distance4', 'distance5'
                                      ,'distance6', 'distance7', 'distance8', 'distance9', 'distance10', 'distance11', 'distance12', 'distance13'
                                      , 'distance14', 'distance15', 'distance16', 'distance17', 'distance18'])
#rename to the filename of clip
df.to_csv('test.csv',index=False)

cap.release()
