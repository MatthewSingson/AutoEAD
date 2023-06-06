import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture("Room 010_segment 20 pose 2.mp4")
with mp_holistic.Holistic(min_detection_confidence  = 0.5, min_tracking_confidence = 0.5) as holistic:
    while True:
        ret, image = cap.read()
        if ret is not True:
            break
        height, width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Facial landmarks
        resultFace = face_mesh.process(rgb_image)
        resultHands = hands.process(rgb_image)
        resultHolistic = holistic.process(image)

        height, width, _ = image.shape

        for facial_landmarks in resultFace.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 1, (100, 100, 0), -1)
        
        if resultHands.multi_hand_landmarks:
            for handLms in resultHands.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        
        mpDraw.draw_landmarks(image, resultHolistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        #press escape to close
        if key == 27:
            break

cap.release()
