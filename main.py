import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
#mport pyttsx3
#engine = pyttsx3.init()


ANN_model=tf.keras.models.load_model("Finalised_ANN_Model.h5")
labels=np.load("labels.npy")



# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _,frame = cap.read()
    x, y, c = frame.shape
    # Flip the frame verticall)y
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)  #handslms.landmark ranges from 0 to 1
                lmy = int(lm.y * y)  #lm.x is on horizontal(position) and lm.y on vertical while x and y are width and height

                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            prediction = ANN_model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)  #prediction contains an array of different confidence or probability levels
                                             #argmax returns the index of max confidence value
            className = labels[classID]
    # show the prediction on the frame
    #putText(frame,classname,(x_coordinate,y_cord),font_type,size(1 for defaul),(color_rgb_value),thckness,antialiasing)
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0,0,255), 2, cv2.LINE_AA)    #--10,50 == top left---||1--default size||(0,0,255)--red color||2--thickness||LINE_AA antialiased(smooth)|| 
   
    cv2.putText(frame,"press x to exit",(450,30),cv2.FONT_HERSHEY_DUPLEX, 
                   0.75, (0,0,255), 2, cv2.LINE_AA)
    # Show the final output
    cv2.imshow("HAND GESTURE RECOGNIZER", frame) 
    #text = str(className)
    
    
    #sound = gTTS("A", lang ="en")
    #sound.save(text + ".mp3")

    
    #playsound.playsound(sound)
    #engine.say(className)
    #engine.runAndWait()
    
    if cv2.waitKey(1) == ord('x'):
        break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
