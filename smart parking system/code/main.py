import cv2
import pickle
import cvzone
import numpy as np
import time

# Open the video capture
cap = cv2.VideoCapture(0)

# Load parking positions from the pickle file
with open("realparkpos", "rb") as f:
    posList = pickle.load(f)

# Define width and height of parking spaces
width, height = 100, 50
# Set output dimensions
output_width, output_height = 640, 480

# Initialize a dictionary to store time counts for each parking space
parking_space_timers = {i: None for i in range(len(posList))}

def format_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def checkparkingspace(imgPro):
    spaceCounter = 0
    current_time = time.time()
    for i, pos in enumerate(posList):
        x, y = pos
        
        imgcrop = imgPro[y:y+height, x:x+width]
        count = cv2.countNonZero(imgcrop)
        cvzone.putTextRect(img, str(count), (x, y+height-3), scale=1, thickness=2, offset=0, colorR=(0,0,255))
        
        if count < 120:  # Adjust the threshold here, considering black surfaces
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            if parking_space_timers[i] is not None:
                parking_space_timers[i] = None
                print(f"Space {i+1}: Transitioned from occupied to free")
        else:
            color = (0, 0, 255)
            thickness = 2
            if parking_space_timers[i] is None:
                parking_space_timers[i] = current_time
            else:
                time_elapsed = current_time - parking_space_timers[i]
                formatted_time = format_time(time_elapsed)
                cvzone.putTextRect(img, formatted_time, (x, y), scale=1, thickness=2, offset=0, colorR=(0, 0, 255))
                print(f"Space {i+1}: Occupied for {formatted_time}")
                # Check if parked for more than 1 minute
                if time_elapsed > 5:
                    # Change color to orange
                    color = (0, 165, 255)  # Orange color in BGR format
                    thickness = 2
                    # Draw warning symbol near the parking slot
                    if int(time.time()) % 2 == 0:  # Blinking every 1 second
                        # Draw black-bordered triangle
                        cv2.drawMarker(img, (x+width//2, y+height//2), color=(0, 0, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=36, thickness=4)
                        # Draw colored triangle inside
                        cv2.drawMarker(img, (x+width//2, y+height//2), color=(0, 0, 255), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=30, thickness=2)
                    print(f"Warning: Space {i+1} occupied for more than 1 minute!")
                
        # Draw rectangle for parking space
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)
        
        # Draw slot number to the left of the parking box
        slot_number_text = str(i+1)
        text_size = cv2.getTextSize(slot_number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x - text_size[0] - 10
        text_y = y + height + 25
        cv2.putText(img, slot_number_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display number of free spaces in the left corner with smaller font size
    free_spaces_text = f'Free Spaces: {spaceCounter}/{len(posList)}'
    # Add black background to the text
    text_bg_color = (0, 0, 0)
    text_bg_padding = 5
    text_size = cv2.getTextSize(free_spaces_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_width, text_height = text_size[0], text_size[1]
    cv2.rectangle(img, (20 - text_bg_padding, 15 - text_height - text_bg_padding), (20 + text_width + text_bg_padding, 15 + text_bg_padding), text_bg_color, cv2.FILLED)
    cv2.putText(img, free_spaces_text, (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(free_spaces_text)

while True:
    # Read a frame from the video capture
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break
    
    # Resize the frame to match the output dimensions
    img = cv2.resize(img, (output_width, output_height))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkparkingspace(imgDilate)
    
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()