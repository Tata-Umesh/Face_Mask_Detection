import cv2
import numpy as np
from tensorflow.keras.models import load_model  
from tensorflow.keras.optimizers import Adam     

model = load_model("Face_Mask_Detection.keras")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load face cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    flipped_frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = flipped_frame[y:y+h, x:x+w]
        resized_frame = cv2.resize(face_roi, (128, 128))
        resized_frame = resized_frame.astype("float32") / 255.0
        input_tensor = np.expand_dims(resized_frame, axis=0)
        
        # Make prediction
        y_pred = model.predict(input_tensor, verbose=0)
        yy_pred = np.argmax(y_pred, axis=-1)
        
        if yy_pred[0] == 0:
            text = "With Mask"
            box_color = (0, 255, 0)  # Green for with mask
        else:
            text = "Without Mask"
            box_color = (0, 0, 255)  # Red for without mask
        
        # Draw bounding box
        cv2.rectangle(flipped_frame, (x, y), (x+w, y+h), box_color, 2)
        
        # Put text above the box
        cv2.putText(flipped_frame, text, (x, y-10), font, font_scale, box_color, thickness)

    cv2.imshow('Face Mask Detection', flipped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()