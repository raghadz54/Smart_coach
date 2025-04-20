from flask import Flask, request, jsonify, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import time
import math

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c) in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_frames(exercise):
    cap = cv2.VideoCapture(0)
    start_time = None
    exercise_duration = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        correct = True
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            knee_distance = np.linalg.norm(np.array(left_knee) - np.array(right_knee))
            ankle_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

            if exercise == "squat":
                if knee_angle < 90:
                    correct = False
                    cv2.putText(image, "Go lower!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif knee_angle > 120:
                    correct = False
                    cv2.putText(image, "Raise your body slightly", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if knee_distance < 0.2 or ankle_distance < 0.2:
                    correct = False
                    cv2.putText(image, "Feet too close!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif knee_distance > 0.4 or ankle_distance > 0.4:
                    correct = False
                    cv2.putText(image, "Feet too wide!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif exercise == "pushup":
                if elbow_angle > 160:
                    correct = False
                    cv2.putText(image, "Lower your body more", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif hip_angle < 160:
                    correct = False
                    cv2.putText(image, "Keep your body straight", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif exercise == "arm_workout":
                if left_wrist[1] < left_shoulder[1]:
                    correct = False
                    cv2.putText(image, "Lower your left arm!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if right_wrist[1] < right_shoulder[1]:
                    correct = False
                    cv2.putText(image, "Lower your right arm!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif exercise == "side_plank":
                foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                plank_angle = calculate_angle(head, left_hip, foot)

                if 170 <= plank_angle <= 180:
                    correct = True
                    if start_time is None:
                        start_time = time.time()
                    elapsed_time = int(time.time() - start_time)
                    remaining_time = max(0, exercise_duration - elapsed_time)
                    cv2.putText(image, f"Time Left: {remaining_time}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    if remaining_time == 0:
                        cv2.putText(image, "Exercise Complete!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        start_time = None
                else:
                    correct = False
                    start_time = None
                    cv2.putText(image, "Adjust your position!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_exercise', methods=['POST'])
def start_exercise():
    data = request.json
    exercise = data.get('exercise', '').lower().replace(" ", "_")
    return jsonify({'message': 'Exercise started'}), 200

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    return Response(generate_frames(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)