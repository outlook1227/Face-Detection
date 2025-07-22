import cv2
import tensorflow as tf
from datetime import datetime  # Import datetime for timestamps
import requests

# List of labels for the model
# The labels should match the order in which the classes were trained in the model
labels = ['Ariel_Sharon', 'Colin_Powell', 'Donald_Rumsfeld', 'George_W_Bush', 'Gerhard_Schroeder', 
          'Hugo_Chavez', 'Jacques_Chirac', 'Jean_Chretien', 'John_Ashcroft', 'Junichiro_Koizumi', 
          'Serena_Williams', 'Thanh_Tai', 'Tien_Do', 'Tony_Blair']

def draw_ped(img, label, x0, y0, xt, yt, color=(255, 127, 0), text_color=(255, 255, 255)):
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x0, y0 + baseline), (max(xt, x0 + w), yt), color, 2)
    cv2.rectangle(img, (x0, y0 - h), (x0 + w, y0 + baseline), color, -1)
    cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return img

# --------- load Haar Cascade model -------------
face_cascade = cv2.CascadeClassifier('D:\\dissertation\\src\\main\\python\\haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = tf.keras.models.load_model("model(1).keras")
print("[INFO] finish load model...")

cap = cv2.VideoCapture(0)

# Open a log file to record detections
log_file = open("d:\\dissertation\\detections_log.txt", "w")

# Counter to track the number of "Tien_Do" detections
object_count = 0
OBJECT_TARGET = 10  # Target number of "Tien_Do" detections

API_URL = "http://127.0.0.1:5000/api/log"  # API endpoint

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        face_count = 0  # Counter to track the number of processed faces in the current frame
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)
            
            result = model.predict(face_img)
            idx = result.argmax(axis=1).item()  # Convert to scalar integer
            confidence = result.max(axis=1) * 100

            # Check if the detected label is in the labels list
            if confidence > 90:
                if idx < len(labels):
                     label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                else:
                    label_text = "UNKNOWN"
            else:
                label_text = "UNKNOWN"
            
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print the label, confidence, and timestamp to the terminal
            print(f"Detected: {label_text} at {timestamp}")
            
            # Log the detection to the file
            log_file.write(f"{timestamp} - Detected: {label_text}\n")

            # Log the detection to the API
            log_data = {"timestamp": timestamp, "label": label_text}
            response = requests.post(API_URL, json=log_data)
            print(f"API Response: {response.json()}")
            
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))
            
            # Check if the detected label is "Tien_Do"
            if labels[idx] == "Tien_Do" and confidence > 90:
                object_count += 1
                print(f"[INFO] Tien_Do detected {object_count}/{OBJECT_TARGET} times with high confidence.")
                log_file.write(f"{timestamp} - Tien_Do detected {object_count}/{OBJECT_TARGET} times with high confidence.\n")
                
                # Stop processing if the target count is reached
                if object_count >= OBJECT_TARGET:
                    print("[INFO] Target of 10 Tien_Do detections reached.")
                    log_file.write(f"{timestamp} - Target of 10 Tien_Do detections reached.\n")
                    cv2.imshow('Detect Face', frame)
                    cv2.waitKey(0)  # Pause to confirm detection
                    cap.release()
                    cv2.destroyAllWindows()
                    log_file.close()  # Close the log file
                    exit()  # Exit the program after confirmation
            
            face_count += 1  # Increment the counter for each processed face in the current frame
        
        cv2.imshow('Detect Face', frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break  # Exit the loop if 'q' is pressed

cv2.destroyAllWindows()
cap.release()
log_file.close()  # Ensure the log file is closed