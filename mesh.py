import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd

def mesh_test_main(vid_filename):
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils 
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    filename = vid_filename
    cap = cv.VideoCapture(filename + ".mp4")
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

    faceXArr = []
    faceYArr = []

    handsXArr = []
    handsYArr = []

    distance1 = []

    fsFace = 0
    fsHands = 0

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

        else:
            d1 = None
            d2 = None
            d3 = None
            d4 = None
            d5 = None
            d6 = None
            d7 = None
            d8 = None
            d9 = None
            d10 = None
            d11 = None
            d12 = None
            d13 = None
            d14 = None
            d15 = None
            d16 = None
            d17 = None
            d18 = None

        if resultHands.multi_hand_landmarks is not None:
            for handLms in resultHands.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):    
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
                    handsXArr.append(cx)
                    handsYArr.append(cy)
            
            if fsFace % 2 == 0:
                xsq = (handsXArr[4] - handsXArr[20]) ** 2
                ysq = (handsYArr[4] - handsYArr[20]) ** 2
                hd1 = np.sqrt(xsq + ysq)

                xsq = (handsXArr[4] - handsXArr[0]) ** 2
                ysq = (handsYArr[4] - handsYArr[0]) ** 2
                hd2 = np.sqrt(xsq + ysq)

                xsq = (handsXArr[20] - handsXArr[0]) ** 2
                ysq = (handsYArr[20] - handsYArr[0]) ** 2
                hd3 = np.sqrt(xsq + ysq)

                xsq = (handsXArr[8] - handsXArr[12]) ** 2
                ysq = (handsYArr[8] - handsYArr[12]) ** 2
                hd4 = np.sqrt(xsq + ysq)

                xsq = (handsXArr[12] - handsXArr[16]) ** 2
                ysq = (handsYArr[12] - handsYArr[16]) ** 2
                hd5 = np.sqrt(xsq + ysq)
        else:
            hd1 = None
            hd2 = None
            hd3 = None
            hd4 = None
            hd5 = None
        
        lm = resultPose.pose_landmarks
        lmPose = mpPose.PoseLandmark

        if lm is not None:
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)

            midpointX = (l_shldr_x + r_shldr_x) / 2
            midpointY = (l_shldr_y + r_shldr_y) / 2

            if fsFace % 2 == 0:
                xsq = (l_shldr_x - midpointX) ** 2
                ysq = (l_shldr_y - midpointY) ** 2
                sd1 = np.sqrt(xsq + ysq)

                xsq = (r_shldr_x - midpointX) ** 2
                ysq = (r_shldr_y - midpointY) ** 2
                sd2 = np.sqrt(xsq + ysq)
        
        else:
            l_shldr_x = None
            l_shldr_y = None
            r_shldr_x = None
            r_shldr_y = None

                
        fsFace += 1
        faceXArr = []
        faceYArr = []
        handsXArr = []
        handsYArr = []
        poseXArr = []
        poseYArr = []

        
        distance1.append((fsFace,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18, hd1, hd2, hd3, hd4, hd5, sd1, sd2))

        cv.imshow(filename, image)
        key = cv.waitKey(1)
        #press escape to close
        if key == 27:
            break
    df = pd.DataFrame(distance1, columns=['frame', 'distance1', 'distance2', 'distance3', 'distance4', 'distance5'
                                        ,'distance6', 'distance7', 'distance8', 'distance9', 'distance10', 'distance11', 'distance12', 'distance13'
                                        , 'distance14', 'distance15', 'distance16', 'distance17', 'distance18', 'hands_dist1', 'hands_dist2'
                                        , 'hands_dist3', 'hands_dist4', 'hands_dist5', 'shoulder_dist1', 'shoulder_dist2'])
    #rename to the filename of clip
    df.to_csv(filename + '.csv',index=False)

    cap.release()
    return df