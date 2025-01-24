import cv2

def main():
    # Load the Haar Cascade XML files
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Start the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and check for smiles
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of Interest (ROI) for the face
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect smiles within the face
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)

        # Display the frame with detected faces and smiles
        cv2.imshow("Face and Smile Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
