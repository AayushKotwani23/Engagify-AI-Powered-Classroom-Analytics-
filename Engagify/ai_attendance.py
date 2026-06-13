import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from deepface import DeepFace

# Create attendance file if not exists
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Emotion", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

def plot_attendance():
    """ Function to plot the graphical representation of attentiveness. """
    df = pd.read_csv(ATTENDANCE_FILE)

    if df.empty:
        print("No attendance data to display.")
        return
    
    # Count occurrences of each emotion
    emotion_counts = df["Emotion"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(emotion_counts.index, emotion_counts.values, color=["green", "red", "blue", "orange"])
    plt.xlabel("Emotions")
    plt.ylabel("Number of Students")
    plt.title("Student Attentiveness Report")
    plt.xticks(rotation=20)
    plt.show()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        
        try:
            # Detect emotion
            emotion_result = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
            emotion = emotion_result[0]["dominant_emotion"]

            # Mark attendance
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv(ATTENDANCE_FILE)
            df = pd.concat([df, pd.DataFrame([[f"Student {len(df)+1}", emotion, timestamp]], columns=df.columns)])
            df.to_csv(ATTENDANCE_FILE, index=False)

            # Draw rectangle and put emotion label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Student Attentiveness Tracker", frame)

    # Press 'q' to quit and show graph
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Show graphical report after quitting
plot_attendance()
