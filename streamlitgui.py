import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'HI', 27: 'YES', 28: 'I LOVE YOU', 29:'NO', 30:'THANKYOU' , 31:'HOW ARE YOU'
}

def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure that data_aux only contains 42 features
        data_aux = data_aux[:42]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        return frame, predicted_character
    
    return frame, "No hand detected"

st.title("Sign Language Detection")
start_stop = st.button('Start/Stop Webcam')

video_placeholder = st.empty()
prediction_placeholder = st.empty()

if 'running' not in st.session_state:
    st.session_state.running = False

if start_stop:
    st.session_state.running = not st.session_state.running

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam")
            break
        processed_frame, prediction = process_frame(frame)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(processed_frame_rgb, channels="RGB")
        prediction_placeholder.text(f"Prediction: {prediction}")
        if not st.session_state.running:
            break

    cap.release()
else:
    st.write("Click 'Start/Stop Webcam' to begin")

cv2.destroyAllWindows()