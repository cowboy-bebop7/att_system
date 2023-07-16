import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(0)  

known_face_encodings = []
known_face_names = []

images_folder = "images"
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

face_locations = []
face_encodings = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

attendance = {}

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

        if name in known_face_names:
            if name not in attendance:
                attendance[name] = 0
            attendance[name] += 1

        if name not in face_names:
            face_names.append(name)

        if name in known_face_names:
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

# Create a bar chart of attendance
names = list(attendance.keys())
counts = list(attendance.values())

plt.bar(names, counts)
plt.xlabel('Names')
plt.ylabel('Attendance Count')
plt.title('Attendance Chart')
plt.show()
