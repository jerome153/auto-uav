import cv2 as cv
from cv2 import aruco
import numpy as np

#load Intrinsic and Extrinsic camera parameters for the camera
calib_data_path = r"C:\Users\3josh\Downloads\MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

#CAMERA INTRINSIC
cam_mat = calib_data["camMatrix"]

#LENS DISTORTION COEFFICIENTS
dist_coef = calib_data["distCoef"]

#ROTATION VECTORS
r_vectors = calib_data["rVector"]

#TRANSLATION VECTORS
t_vectors = calib_data["tVector"]

#If this is marker size in centimeters, then this should be about 13 inches (12 + 1 frame)
MARKER_SIZE = 33

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

param_markers = cv.aruco.DetectorParameters()


#This is actively converting a streamed image to greyscale, detecting the aruco, pulling corners and ID
# We want to replace this this a saved greyscale image pulled when the correct marker flag is thrown

cap = cv.VideoCapture(0) #give the server id shown in IP webcam App
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # this is how we have it in cv2.py
    # convert the image to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the markers in the frame
    #detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
    #corners, ids, rejected = detector.detectMarkers(gray)
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    marker_corners, marker_IDs, reject = cv.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )


    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()


            
            # calculate the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )



            # for pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()