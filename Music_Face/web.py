import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import random

# Load the trained CNN model
cnn_model = load_model("CNN_Model.keras")

# Load music data
music_data = pd.read_csv("C:/Mukul/Emotion Detection & Music Recommendation Final/Music_Face/spotify-music-data-to-identify-the-moods/data_playlist.csv")

# Define mood-to-music mapping (example)
mood_to_music_mapping = {
    "angry": ["rock", "metal"],
    "disgust": ["classical", "jazz"],
    "fear": ["ambient", "experimental"],
    "happy": ["pop", "dance"],
    "calm": ["pop", "instrumental"],
    "sad": ["blues", "indie"],
    "surprise": ["electronic", "hip-hop"]
}

# Function to preprocess image for model input
def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    img_height = 48
    img_width = 48
    resized_image = cv2.resize(image, (img_width, img_height))
    
    # Convert the image to float32 and rescale the pixel values to [0, 1]
    resized_image = resized_image.astype(np.float32) / 255.0
    
    return resized_image


# Function to make predictions using CNN model
def predict_with_cnn(image):
    image = preprocess_image(image)
    prediction = cnn_model.predict(np.expand_dims(image, axis=0))
    return prediction

# Function to recommend music based on mood
def recommend_music(predicted_mood):
    predicted_mood = predicted_mood.capitalize()  # Capitalize the predicted mood
    if predicted_mood in music_data['mood'].unique():
        recommended_music = music_data[music_data['mood'] == predicted_mood]
        return recommended_music[['name', 'playlist_link']].sample(1).values  # Sample 1 random music tracks with name and link
    else:
        return None

# Streamlit UI components
st.title("Emotion Detection and Music Recommendation")
st.write("This project employs advanced emotion recognition algorithms to analyze your facial expressions to determine your current mood accurately and recommend music with respect to your moods.")

# Adjust the size of the "Detect Emotion" button and center-align
start_detection = st.button(label="Start Detection", help="Press to start emotion detection", key="start_detection")

# OpenCV code to capture video from webcam
cap = cv2.VideoCapture(0)  # Use default webcam (index 0)

# Variables to control the detection and recommendation process
stop_detection = False
time_elapsed = 0
detected_emotion = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if start_detection:
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face region
            face_region = frame[y:y+h, x:x+w]

            # Make predictions with the CNN model
            prediction = predict_with_cnn(face_region)

            # Extract predicted mood from the prediction (example: argmax)
            predicted_mood_index = np.argmax(prediction)
            predicted_mood = ["angry", "disgust", "fear", "happy", "calm", "sad", "surprise"][predicted_mood_index]
            
            detected_emotion = predicted_mood

            # Display predicted mood
            cv2.putText(frame, f"Mood: {predicted_mood}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame with rectangles and predicted mood
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_detection:
        break

    # Update time elapsed
    time_elapsed += 1

    # Break the loop if 20 seconds have elapsed
    if time_elapsed >= 20:
        stop_detection = True

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Display detected emotion and recommended songs
if detected_emotion:
    st.subheader(f"Your Detected Emotion is: {detected_emotion.capitalize()}")
    recommended_songs = recommend_music(detected_emotion)
    if recommended_songs is not None:
        st.subheader("Recommended songs for your mood are:")
        for song in recommended_songs:
            st.markdown(f"[{song[0]}]({song[1]})")  # Display name as hyperlink to the provided link
    else:
        st.subheader("No music recommendations for this mood.")
else:
    st.write("No emotion detected.")
