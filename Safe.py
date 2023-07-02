import cv2
import time

def detect_eyes():
    # Load the pre-trained eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Start video capture
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    eye_detected = False

    while True:
        # Read the video frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes in the grayscale frame
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if eyes are detected
        if len(eyes) > 0:
            eye_detected = True

        # Draw rectangles around the detected eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Eyes Detection', frame)

        # Check if eye has been detected for more than 30 seconds
        elapsed_time = time.time() - start_time
        if eye_detected and elapsed_time >= 30:
            print('Hello')
            break

        # Check for 'q' key press to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Call the function to start eye detection
detect_eyes()
