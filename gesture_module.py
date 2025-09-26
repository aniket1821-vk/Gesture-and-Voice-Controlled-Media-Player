# gesture_module.py
import cv2
import numpy as np
import math

class GestureRecognition:
    def __init__(self):
        pass

    def detect_gesture(self, contour):
        # Placeholder for number of fingers detected
        num_fingers = 0

        # Find convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate the angle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / math.pi

                # If angle is less than 90 degrees, consider it as a finger
                if angle <= 90:
                    num_fingers += 1

        # Determine gesture based on number of fingers
        if num_fingers == 0:
            return "Play/Pause"
        elif num_fingers == 1:
            return "Next Track"
        elif num_fingers == 2:
            return "Previous Track"
        elif num_fingers == 3:
            return "Volume Up"
        elif num_fingers == 4:
            return "Volume Down"
        else:
            return None
