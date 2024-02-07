import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reference_image = cv2.imread('me.jpg', cv2.IMREAD_GRAYSCALE)

# Function to check if the reference face is present in the current frame
def is_reference_face_present(frame):
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))


    
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for face matching
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Match the reference face with the detected face
        result = cv2.matchTemplate(roi_gray, reference_image, cv2.TM_CCOEFF_NORMED)
        
        # Define a threshold for matching
        threshold = 0.8
        
        # Check if the match is above the threshold
        if result.max() > threshold:
            return True  # Reference face found in the current frame
    
    return False  # Reference face not found


cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the reference face is present in the current frame
    if is_reference_face_present(frame):
        cv2.putText(frame, 'Reference Face Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Live Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
