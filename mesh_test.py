import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd

mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 
mpPose = mp.solutions.pose
pose = mpPose.Pose()

filename = "r02 - 02.1e"

cap = cv.VideoCapture(filename + ".mp4")
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

fsFace = 0
fsHands = 0

print(f"Height : {height} || Width : {width}")

faceXArr = []
faceYArr = []

handsXArr = []
handsYArr = []

poseXArr = []
poseYArr = []


distance1 = []

while True:
    ret, image = cap.read()
    if ret is not True:
        break
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_image)
    resultHands = hands.process(rgb_image)
    resultPose = pose.process(rgb_image)

    if result.multi_face_landmarks is not None:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range (0, 468):
                point = facial_landmarks.landmark[i]
                pX = int(point.x * width)
                pY = int(point.y * height)
                faceYArr.append(pY)
                faceXArr.append(pX)
                cv.circle(image, (pX, pY), 3, (100, 100, 0), -1)
            
        if fsFace % 2 == 0:

            xsq = (faceXArr[12] - faceXArr[15]) ** 2
            ysq = (faceYArr[12] - faceYArr[15]) ** 2
            d1 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[11] - faceXArr[17]) ** 2
            ysq = (faceYArr[11] - faceYArr[17]) ** 2
            d2 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[81] - faceXArr[181]) ** 2
            ysq = (faceYArr[81] - faceYArr[181]) ** 2
            d3 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[312] - faceXArr[314]) ** 2
            ysq = (faceYArr[312] - faceYArr[314]) ** 2
            d4 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[83] - faceXArr[175]) ** 2
            ysq = (faceYArr[83] - faceYArr[175]) ** 2
            d5 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[314] - faceXArr[176]) ** 2
            ysq = (faceYArr[314] - faceYArr[176]) ** 2
            d6 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[314] - faceXArr[377]) ** 2
            ysq = (faceYArr[314] - faceYArr[377]) ** 2
            d7 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[405] - faceXArr[400]) ** 2
            ysq = (faceYArr[405] - faceYArr[400]) ** 2
            d8 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[320] - faceXArr[377]) ** 2
            ysq = (faceYArr[320] - faceYArr[377]) ** 2
            d9 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[335] - faceXArr[377]) ** 2
            ysq = (faceYArr[335] - faceYArr[377]) ** 2
            d10 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[426] - faceXArr[430]) ** 2
            ysq = (faceYArr[426] - faceYArr[430]) ** 2
            d11 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[32] - faceXArr[216]) ** 2
            ysq = (faceYArr[32] - faceYArr[216]) ** 2
            d12 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[352] - faceXArr[152]) ** 2
            ysq = (faceYArr[352] - faceYArr[152]) ** 2
            d13 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[123] - faceXArr[152]) ** 2
            ysq = (faceYArr[123] - faceYArr[152]) ** 2
            d14 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[427] - faceXArr[365]) ** 2
            ysq = (faceYArr[427] - faceYArr[365]) ** 2
            d15 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[216] - faceXArr[170]) ** 2
            ysq = (faceYArr[216] - faceYArr[170]) ** 2
            d16 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[58] - faceXArr[148]) ** 2
            ysq = (faceYArr[58] - faceYArr[148]) ** 2
            d17 = np.sqrt(xsq + ysq)

            xsq = (faceXArr[291] - faceXArr[379]) ** 2
            ysq = (faceYArr[291] - faceYArr[379]) ** 2
            d18 = np.sqrt(xsq + ysq)

    if resultHands.multi_hand_landmarks is not None:
        for handLms in resultHands.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):    
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                handsXArr.append(cx)
                handsYArr.append(cy)
        
        if fsFace % 2 == 0:
            xsq = (handsXArr[4] - faceXArr[20]) ** 2
            ysq = (faceYArr[4] - faceYArr[20]) ** 2
            hd1 = np.sqrt(xsq + ysq)
    else:
        hd1 = None

            
    fsFace += 1
    faceXArr = []
    faceYArr = []
    handsXArr = []
    handsYArr = []
    poseXArr = []
    poseYArr = []

    
    distance1.append((fsFace,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18, hd1))

    cv.imshow(filename, image)
    key = cv.waitKey(1)
    #press escape to close
    if key == 27:
        break

df = pd.DataFrame(distance1, columns=['frame', 'distance1', 'distance2', 'distance3', 'distance4', 'distance5'
                                      ,'distance6', 'distance7', 'distance8', 'distance9', 'distance10', 'distance11', 'distance12', 'distance13'
                                      , 'distance14', 'distance15', 'distance16', 'distance17', 'distance18', 'hands_dist1'])
#rename to the filename of clip
df.to_csv(filename + '-test.csv',index=False)

cap.release()
