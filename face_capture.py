import os
import cv2
import requests

dataset_folder = r"D:\Face_Detection\dataset"  # Use raw string for consistent path handling

cap = cv2.VideoCapture(0)

my_name = "Thanh_Tai"
person_folder = os.path.join(dataset_folder, my_name)  # Construct the full path
os.makedirs(person_folder, exist_ok=True)  # Ensure the folder exists
num_sample = 70

API_URL = "http://127.0.0.1:5000/api/log"  # API endpoint

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        cv2.imshow("Capture Photo", frame)
        file_path = os.path.join(person_folder, f"{my_name}_{i:04d}.jpg")  # Construct file path
        cv2.imwrite(file_path, cv2.resize(frame, (250, 250)))

        # Send captured face data to the API
        response = requests.post(API_URL, json={"name": my_name, "image_path": file_path})
        print(f"API Response: {response.json()}")
        
        if cv2.waitKey(800) == ord('q') or i == num_sample:
            break
        i += 1    

cap.release()
cv2.destroyAllWindows()