import cv2
import dlib
import pandas as pd

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("oculAI/face_features/shape_predictor_68_face_landmarks.dat")

# read the image
#This line for getting the image from your webcam
cap = cv2.VideoCapture(0)
#This line for getting the image from a saved image
img = cv2.imread("trump.jpeg")

cols = [str(n) for n in range(0, 68)]
df_points = pd.DataFrame(columns=cols)

frame = 0

while True:
    frame += 1
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        face_featu_dict = {}
        # Loop through all the points

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # Saving the coordinates from all the points in a dictionary

            face_featu_dict[str(n)] = landmarks.part(n)
        #adding a row with the info about all the points
        df_points = df_points.append(pd.DataFrame.from_records([face_featu_dict]), ignore_index=True)



    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        #saving the df with all the info about the points
        df_points.to_csv('oculAI/face_features/output_csv/face_features.csv')
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()