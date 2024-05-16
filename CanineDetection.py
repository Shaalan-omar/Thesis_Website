# coding: utf-8

# In[13]:
# In[14]:


import glob
import cv2
import numpy as np
import os.path
from roboflow import Roboflow
import matplotlib.pyplot as plt
import csv
import os
import sys
import json


# In[15]:


# Initialize Roboflow client
rf = Roboflow(api_key="5ZFtB1hrrQTykeWZrSWd")
    
# Load the Roboflow workspace and project
workspace = rf.workspace()
project = workspace.project("right-labeling")
    
# Select the model version
model_version = 2
    
# Load the model from the project
model_R= project.version(model_version).model

project2 = workspace.project("left-labeling")
    
# Select the model version
model_version2 = 4
    
# Load the model from the project
model_L=project2.version(model_version2).model


# In[16]:


def process_image(image_path):
    angleR=''
    verticalR=''
    overR=''
    apexR=''
    angleL=''
    verticalL=''
    overL=''
    apexL=''
    impactL=True
    impactR=True
    
    #preprocessing 
    preproc(image_path)

    #yolo
    image_name = os.path.basename(image_path)
    image_name = image_name.split('.')[0]
    original_image=yolo(image_name)
    output_folder = "static/processed"
    output_path = os.path.join(output_folder, f'processed_{image_name}.jpg')
    cv2.imwrite(output_path, original_image)

    if original_image is not None:
        #vertical height 
        impactL,impactR,verticalR,verticalL= vertHe(original_image,impactL,impactR,verticalR,verticalL)

        #angulation
        angleR,angleL= angl(original_image,impactL,impactR,angleR,angleL)
        
        #Overlap
        overR,overL=over(original_image,impactL,impactR,overR,overL)
    
        #Apex Position
        apexR,apexL= apex(original_image,impactL,impactR,apexR,apexL)
   
    return(impactL,impactR,angleR,verticalR,overR,apexR,angleL,verticalL,overL,apexL)


# In[17]:


def preproc(image_path):
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    #resizing
    resized = cv2.resize(img, (780, 540), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite('resized.jpg',resized)


# In[18]:


def yolo(image_num):
    original_image=None
    
   # Define the color for each class
    class_colors = {
        "L_lateral-incisor": (255, 0, 0),  # Blue
        "L_Canine": (0, 255, 0),            # Green
        "L_first-premolar": (0, 0, 255),    # Red
        "L_second-premolar": (0, 255, 255),  # Yellow
        
        "R_lateral incisor": (255, 255, 0),  # Cyan
        "R_Canine": (255, 0, 255),            # Pink
        "R_first premolar": (128, 0, 128),    # Purple
        "R_second premolar": (0, 165, 255)  # Orange
    }
    
    # Load the input image
    input_image_path ='resized.jpg'
    input_image = cv2.imread(input_image_path)
    
    # Make predictions for Right  on the input image
    if model_R is None and model_L is None:
        print(image_num)
        image_path = f"static/uploads/YOLO/{image_num}_YOLO.jpg"
        original_image = cv2.imread(image_path)
        return original_image
    else:
        predictions = model_R.predict(input_image_path, confidence=10).json()
        # Loop through the predictions and draw colored segmentations
        for prediction in predictions["predictions"]:
            class_name = prediction["class"]
            if class_name=="R_Canine" :
                continue 
                
            color= None
            points = prediction["points"]
            
            # Check x-axis values
            x_values = [point["x"] for point in points]
            
            if all(x < 390 for x in x_values) and all(x > 100 for x in x_values):
                if class_name in ["R_lateral incisor","R_first premolar", "R_second premolar"]:
                    color = class_colors[class_name]
        
            # Convert points to numpy array of integers
            points_np = [(int(point["x"]), int(point["y"])) for point in points]
            points_np = np.array(points_np)

            if color is not None : 
                # Draw filled polygon
                original_image=cv2.fillPoly(input_image, [points_np], color)


        # Loop through the predictions and draw canine
        for prediction in predictions["predictions"]:
            class_name = prediction["class"]
            if class_name=="R_Canine" :          
                color= None
                points = prediction["points"]
                
                # Check x-axis values
                x_values = [point["x"] for point in points]
                
                if all(x < 390 for x in x_values) and all(x > 100 for x in x_values):
                    color = class_colors[class_name]
            
                # Convert points to numpy array of integers
                points_np = [(int(point["x"]), int(point["y"])) for point in points]
                points_np = np.array(points_np)
        
                if color is not None : 
                    # Draw filled polygon
                    original_image=cv2.fillPoly(input_image, [points_np], color)

    
        # Make predictions for Left  on the input image
        predictions = model_L.predict(input_image_path,confidence=10).json()
        # Loop through the predictions and draw colored segmentations
        for prediction in predictions["predictions"]:
            class_name = prediction["class"]
            if class_name=="L_Canine" :
                continue 
            color= None
            points = prediction["points"]
            
            # Check x-axis values
            x_values = [point["x"] for point in points]
            if all(x > 390 for x in x_values):
                if class_name in ["L_lateral-incisor", "L_first-premolar", "L_second-premolar"]:
                    color = class_colors[class_name]
        
            # Convert points to numpy array of integers
            points_np = [(int(point["x"]), int(point["y"])) for point in points]
            points_np = np.array(points_np)

            if color is not None : 
                # Draw filled polygon
                original_image=cv2.fillPoly(original_image, [points_np], color)

        #color canine
        for prediction in predictions["predictions"]:
            class_name = prediction["class"]
            if class_name=="L_Canine" :
                color= None
                points = prediction["points"]     
                # Check x-axis values
                x_values = [point["x"] for point in points]
                if all(x > 390 for x in x_values):
                    color = class_colors[class_name]
            
                # Convert points to numpy array of integers
                points_np = [(int(point["x"]), int(point["y"])) for point in points]
                points_np = np.array(points_np)
        
                if color is not None : 
                    # Draw filled polygon
                    original_image=cv2.fillPoly(original_image, [points_np], color)

        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off') # this will hide the x-axis and y-axis
        # plt.show()

        cv2.imwrite('4-labeled.jpg',original_image)
        return original_image 


# In[19]:


def vertHe(original_image, impactL, impactR, verticalR, verticalL):
    image = original_image.copy()

    # Detecting the left canine and drawing the ellipse
    left_color = np.array([0, 255, 0])  # Green
    color_tolerance = 40
    lower_left = left_color - color_tolerance
    upper_left = left_color + color_tolerance
    mask = cv2.inRange(image, lower_left, upper_left)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lower_tip_y = None

    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        lowest_point = None
        highest_point = None

        for point in main_contour:
            x, y = point[0]

            if lowest_point is None or y > lowest_point[1]:
                lowest_point = (x, y)

            if highest_point is None or y < highest_point[1]:
                highest_point = (x, y)

        if lowest_point:
            cv2.circle(image, lowest_point, 3, (100, 100, 192), -1) 

        if highest_point:
            cv2.circle(image, highest_point, 3, (100, 100, 192), -1)

        lower_tip_y = lowest_point[1]

    # Detecting the blue bounding box
    blue_color = np.array([255, 0, 0])
    color_tolerance = 25
    mask_blue = cv2.inRange(image, blue_color - color_tolerance, blue_color + color_tolerance)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_blue:
        largest_contour_blue = max(contours_blue, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour_blue)
        if h <  80:
            y = y - (80 - h)
            h = 80
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)

            lower_boundary_y = y + h
            height = lower_boundary_y - y  
            first_section_height = int(0.34 * height)
            middle_section_height = int(0.58 * height)
            last_section_height = int(0.08 * height)
            mid1_y = y + first_section_height  
            mid2_y = mid1_y + middle_section_height  
            top_boundary_y = y  

            cv2.line(image, (x, lower_boundary_y), (x + w, lower_boundary_y), (255, 0, 0), 2)  
            cv2.line(image, (x, mid1_y), (x + w, mid1_y), (0, 0, 255), 2)  
            cv2.line(image, (x, mid2_y), (x + w, mid2_y), (0, 255, 255), 2)  
            cv2.line(image, (x, top_boundary_y), (x + w, top_boundary_y), (255, 255, 0), 2)

            if lower_tip_y is not None:
                if lower_tip_y >= mid2_y:
                    impactL = False
                elif lower_tip_y < mid2_y and lower_tip_y >= mid1_y-4:
                    verticalL = "Good"
                elif lower_tip_y < mid1_y-4 and lower_tip_y > top_boundary_y+2:
                    verticalL = "Average"
                elif lower_tip_y <= top_boundary_y+2:
                    verticalL = "Poor"
        else:
                
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)

            lower_boundary_y = y + h
            height = lower_boundary_y - y  
            first_section_height = int(0.34 * height)
            middle_section_height = int(0.58 * height)
            last_section_height = int(0.08 * height)
            mid1_y = y + first_section_height  
            mid2_y = mid1_y + middle_section_height  
            top_boundary_y = y  

            cv2.line(image, (x, lower_boundary_y), (x + w, lower_boundary_y), (255, 0, 0), 2)  
            cv2.line(image, (x, mid1_y), (x + w, mid1_y), (0, 0, 255), 2)  
            cv2.line(image, (x, mid2_y), (x + w, mid2_y), (0, 255, 255), 2)  
            cv2.line(image, (x, top_boundary_y), (x + w, top_boundary_y), (255, 255, 0), 2)

            if lower_tip_y is not None:
                if lower_tip_y >= mid2_y:
                    impactL = False
                elif lower_tip_y < mid2_y and lower_tip_y >= mid1_y-1:
                    verticalL = "Good"
                elif lower_tip_y < mid1_y-1 and lower_tip_y > top_boundary_y+2:
                    verticalL = "Average"
                elif lower_tip_y <= top_boundary_y+2:
                    verticalL = "Poor"

    right_color = np.array([255, 0, 255])  
    color_tolerance2 = 25
    mask_right = cv2.inRange(image, right_color - color_tolerance2, right_color + color_tolerance2)
    contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lower_tip_y = None

    if contours_right:
        main_contour = max(contours_right, key=cv2.contourArea)

        lowest_point = None
        highest_point = None

        for point in main_contour:
            x, y = point[0]

            if lowest_point is None or y > lowest_point[1]:
                lowest_point = (x, y)

            if highest_point is None or y < highest_point[1]:
                highest_point = (x, y)

        if lowest_point:
            cv2.circle(image, lowest_point, 3, (100, 100, 192), -1)

        if highest_point:
            cv2.circle(image, highest_point, 3, (100, 100, 192), -1)

        lower_tip_y = lowest_point[1]

    cayan_color = np.array([255, 255, 0])
    color_tolerance = 40
    mask_cayan = cv2.inRange(image, cayan_color - color_tolerance, cayan_color + color_tolerance)
    contours_cayan, _ = cv2.findContours(mask_cayan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_cayan:
        largest_contour_cayan = max(contours_cayan, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour_cayan)
        if h <  80:
            y = y - (80 - h)
            h = 80
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            
            lower_boundary_y = y + h
            height = lower_boundary_y - y  
            first_section_height = int(0.36 * height)
            middle_section_height = int(0.48 * height)
            last_section_height = int(0.34 * height)
            mid1_y = y + first_section_height  
            mid2_y = mid1_y + middle_section_height  
            top_boundary_y = y  

            cv2.line(image, (x, lower_boundary_y), (x + w, lower_boundary_y), (255, 0, 0), 2)  
            cv2.line(image, (x, mid1_y), (x + w, mid1_y), (0, 0, 255), 2)  
            cv2.line(image, (x, mid2_y), (x + w, mid2_y), (0, 255, 255), 2)  
            cv2.line(image, (x, top_boundary_y), (x + w, top_boundary_y), (255, 255, 0), 2) 
            
            if lower_tip_y is not None:
                if lower_tip_y >= mid2_y:
                    impactR = False
                elif lower_tip_y < mid2_y and lower_tip_y >= mid1_y-4:
                    verticalR = "Good"
                elif lower_tip_y < mid1_y-4 and lower_tip_y >= top_boundary_y:
                    verticalR = "Average"
                elif lower_tip_y < top_boundary_y:
                    verticalR = "Poor"
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            
            lower_boundary_y = y + h
            height = lower_boundary_y - y  
            first_section_height = int(0.3 * height)
            middle_section_height = int(0.48 * height)
            last_section_height = int(0.34 * height)
            mid1_y = y + first_section_height
            mid2_y = mid1_y + middle_section_height  
            top_boundary_y = y  

            cv2.line(image, (x, lower_boundary_y), (x + w, lower_boundary_y), (255, 0, 0), 2)  
            cv2.line(image, (x, mid1_y), (x + w, mid1_y), (0, 0, 255), 2)  
            cv2.line(image, (x, mid2_y), (x + w, mid2_y), (0, 255, 255), 2)  
            cv2.line(image, (x, top_boundary_y), (x + w, top_boundary_y), (255, 255, 0), 2) 
            
            if lower_tip_y is not None:
                if lower_tip_y >= mid2_y:
                    impactR = False
                elif lower_tip_y < mid2_y and lower_tip_y >= mid1_y-2:
                    verticalR = "Good"
                elif lower_tip_y < mid1_y-2 and lower_tip_y >= top_boundary_y+1:
                    verticalR = "Average"
                elif lower_tip_y < top_boundary_y+1:
                    verticalR = "Poor"


    return impactL, impactR, verticalR, verticalL


# In[20]:


def angl(original_image,impactL,impactR,angleR,angleL):
    image =original_image.copy()
    # Define the tolerance range for the color (adjustable)
    color_tolerance = 40
    color_tolerance2 = 25
    
    #Left Canine 
    if impactL :
        left_color = np.array([0,255, 0])  # BGR format Green
        lower_left = left_color - color_tolerance
        upper_left = left_color + color_tolerance
        mask= cv2.inRange(image, lower_left, upper_left)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If contours are found
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Fit an ellipse to the contour
            if largest_contour.size > 10:
                
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Extract the angle of inclination from the fitted ellipse
                angle = ellipse[2]
                if angle >90 :
                    angle=180-angle
                
                # # Display the angle
                # print(f"Angle of inclination: {angle} degrees")
                center_x, center_y = map(int, ellipse[0])
                
                # Draw the ellipse on the original image
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                
                # Calculate the endpoints for the main axis of the ellipse
                main_axis_length = int(ellipse[1][1] / 2)  # Half the length of minor axis
                x_main = int(ellipse[0][0] - main_axis_length * np.sin(np.radians(ellipse[2])))
                y_main = int(ellipse[0][1] + main_axis_length * np.cos(np.radians(ellipse[2])))
        
                # Draw the main axis of the ellipse
                cv2.line(image, (int(ellipse[0][0] + main_axis_length * np.sin(np.radians(ellipse[2]))), 
                                int(ellipse[0][1] - main_axis_length * np.cos(np.radians(ellipse[2])))),
                        (x_main, y_main), (255, 0, 0), 2)
                    # Draw the y-axis from the center of the ellipse
                cv2.line(image, (center_x, 10), (center_x, image.shape[0]), (255, 255, 255), 2)
        
                 # Write the angle text on the image
                y_axis_height = 300  # Change this value to your desired height
                y_axis_top = max(0, center_y - (y_axis_height // 2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 255, 255)  # White color
                font_thickness = 2
                angle_text = f"Angle: {angle.__round__(3)} degrees"
                text_size = cv2.getTextSize(angle_text, font, font_scale, font_thickness)[0]
                text_x = center_x + 70
                text_y = center_y - 75
        
                cv2.putText(image, angle_text, (text_x, text_y), font, font_scale, font_color, font_thickness)
                if angle<=(10):
                    angleL="Good"
                elif angle>(10) and angle<=(25):
                    angleL="Average"
                elif angle>(25):
                    angleL="Poor"
    
    #Right Canine 
    if impactR : 
        right_color = np.array([255, 0, 255])  # BGR format Pink
        lower_right = right_color - color_tolerance2
        upper_right = right_color + color_tolerance2
        mask2= cv2.inRange(image, lower_right, upper_right)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            # Get the largest contour
            largest_contour = max(contours2, key=cv2.contourArea)
            # Fit an ellipse to the contour
            if largest_contour.size > 10:
                
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Extract the angle of inclination from the fitted ellipse
                angle = ellipse[2]
                
                angle = 180 - angle
                if angle >90 :
                    angle=180-angle
                # # Display the angle
                # print(f"Angle of inclination: {angle} degrees")
                center_x, center_y = map(int, ellipse[0])
                
                # # Draw the ellipse on the original image
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
        
        
                # Calculate the endpoints for the main axis of the ellipse
                main_axis_length = int(ellipse[1][1] / 2)  # Half the length of minor axis
                x_main = int(ellipse[0][0] - main_axis_length * np.sin(np.radians(ellipse[2])))
                y_main = int(ellipse[0][1] + main_axis_length * np.cos(np.radians(ellipse[2])))
        
                # Draw the main axis of the ellipse
                cv2.line(image, (int(ellipse[0][0] + main_axis_length * np.sin(np.radians(ellipse[2]))), 
                                int(ellipse[0][1] - main_axis_length * np.cos(np.radians(ellipse[2])))),
                        (x_main, y_main), (255, 0, 0), 2)
                
                         # Write the angle text on the image
                y_axis_height = 300  # Change this value to your desired height
                y_axis_top = max(0, center_y - (y_axis_height // 2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 255, 255)  # White color
                font_thickness = 2
                angle_text = f"Angle: {angle.__round__(3)} degrees"
                text_size = cv2.getTextSize(angle_text, font, font_scale, font_thickness)[0]
                text_x = center_x - 200
                text_y = center_y - 80
                cv2.putText(image, angle_text, (text_x, text_y), font, font_scale, font_color, font_thickness)
                if angle<=(10):
                    angleR="Good"
                elif angle>(10) and angle<=(25):
                    angleR="Average"
                elif angle>(25):
                    angleR="Poor"

    return(angleR,angleL)


# In[21]:


def over(original_image,impactL,impactR,overR,overL):
    image =original_image.copy()
    # Convert the image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Right_canine_tip_x=None
    Left_canine_tip_x=None

    if impactL :
        rightmost_x=None
        rightmost_point = None
        # Define the range for blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Define the range for green color in HSV
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        ####################
        Right_canine_tip_x = None


        if green_contours:
            main_contour = max(green_contours, key=cv2.contourArea)

            # Initialize the lowest and highest points
            lowest_point = None
            highest_point = None
            lowest_x = None
            lowest_y = None

            # Loop through the contour points
            for point in main_contour:
                x, y = point[0]

                 # Check if this point has a lower x value
                if lowest_y is None or y > lowest_y:
                    lowest_x = x
                    lowest_y = y
                    lowest_point = (lowest_x, lowest_y)
                # Check if this point has the same x value and a lower y value
                elif y == lowest_y and (lowest_x is None or x < lowest_x):
                    lowest_x = x
                    lowest_point = (lowest_x, lowest_y)


            # Draw the points on the original image
            if lowest_point:
                cv2.circle(image, lowest_point, 5, (0, 0, 255), -1)  # Draw in red


            Left_canine_tip_x = lowest_point[0]
            
            # Check if Rightmost point is found

                ###########################
            for contour in blue_contours:
                if cv2.contourArea(contour) > 100:  # Filter out small contours
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(image, ellipse, (255, 0, 0), 2)  # Blue ellipse
                    Left_center_x, Left_center_y = map(int, ellipse[0])
                    center = ellipse [0]
                    axes = ellipse[1]
                    angle = ellipse[2]
                    angle_rad = np.radians(angle - 90)
                    major_axis_length = max(axes) / 2
                    rightmost_x = center[0] + major_axis_length * np.cos(angle_rad)


            # Calculate the endpoints for the main axis of the ellipse
                    Left_main_axis_length = int(ellipse[1][1] / 2)  # Half the length of the minor axis
                    Left_x_main = int(Left_center_x - Left_main_axis_length * np.sin(np.radians(ellipse[2])))
                    Left_y_main = int(Left_center_y + Left_main_axis_length * np.cos(np.radians(ellipse[2])))

                    # Draw the main axis of the ellipse
                    cv2.line(image, (int(ellipse[0][0] + Left_main_axis_length * np.sin(np.radians(ellipse[2]))), 
                                int(ellipse[0][1] - Left_main_axis_length * np.cos(np.radians(ellipse[2])))),
                        (Left_x_main, Left_y_main), (0, 255, 0), 2)

        ###################
               
        if Left_canine_tip_x is not None and rightmost_x is not None and Left_center_x is not None: 
            if Left_canine_tip_x - 1>= rightmost_x:
                overL = "Good"
            elif rightmost_x > Left_canine_tip_x - 1 > Left_center_x:
                overL = "Average"
            elif Left_canine_tip_x - 1 <= Left_center_x:
                overL = "Poor"
    if impactR:
        #pink
        leftmost_x=None
        leftmost_point=None
        pink_color=np.array([255, 0, 255])  # BGR format Pink
        color_tolerance = 40
        color_tolerance2 = 25
        pink_lower = pink_color - color_tolerance2
        pink_upper = pink_color + color_tolerance2
        pink_mask = cv2.inRange(image, pink_lower, pink_upper)
        pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cyan
        cyan_lower = np.array([85, 100, 100])
        cyan_upper = np.array([95, 255, 255])
        cyan_mask = cv2.inRange(hsv_image, cyan_lower, cyan_upper)
        cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        Right_canine_tip_x = None
        if pink_contours :
            main_contour = max(pink_contours, key=cv2.contourArea)

            # Initialize the lowest and highest points
            lowest_point = None
            highest_point = None
            lowest_x = None
            lowest_y = None

            # Loop through the contour points
            for point in main_contour:
                x, y = point[0]

                 # Check if this point has a lower x value
                if lowest_y is None or y > lowest_y:
                    lowest_x = x
                    lowest_y = y
                    lowest_point = (lowest_x, lowest_y)
                # Check if this point has the same x value and a lower y value
                elif y == lowest_y and (lowest_x is None or x > lowest_x):
                    lowest_x = x
                    lowest_point = (lowest_x, lowest_y)


            # Draw the points on the original image
            if lowest_point:
                cv2.circle(image, lowest_point, 5, (0, 0, 255), -1)  # Draw in red


            Right_canine_tip_x = lowest_point[0]

         # cyan contours
        for contour in cyan_contours:
            if cv2.contourArea(contour) > 100:  # Filter out small contours
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(image, ellipse, (255, 0, 0), 2)  # Blue ellipse
                Right_center_x,Right_center_y = map(int, ellipse[0])
                #######
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]
                angle_rad = np.radians(angle - 90)
                major_axis_length = max(axes) / 2
                leftmost_x = center[0] - major_axis_length * np.cos(angle_rad)

             # Calculate the endpoints for the main axis of the ellipse
                Right_main_axis_length = int(ellipse[1][1] / 2)  # Half the length of the minor axis
                Right_x_main = int(Right_center_x - Right_main_axis_length * np.sin(np.radians(ellipse[2])))
                Right_y_main = int(Right_center_y + Right_main_axis_length * np.cos(np.radians(ellipse[2])))

                # Draw the main axis of the ellipse
                cv2.line(image, (int(ellipse[0][0] + Right_main_axis_length * np.sin(np.radians(ellipse[2]))), 
                            int(ellipse[0][1] - Right_main_axis_length * np.cos(np.radians(ellipse[2])))),
                    (Right_x_main, Right_y_main), (0, 255, 0), 2)

        if Right_canine_tip_x is not None and leftmost_x is not None and Right_center_x is not None:
            if (Right_canine_tip_x + 1 ) <= leftmost_x:
                overR = "Good"
            elif  leftmost_x < (Right_canine_tip_x + 1 ) < Right_center_x:
                overR = "Average"
            elif (Right_canine_tip_x + 1 ) >= Right_center_x :
                overR = "Poor"
         
    return(overR,overL)


# In[22]:


def apex(or_image,impactL,impactR,apexR,apexL):
    input_image=or_image.copy()
    overlay = input_image.copy()
    # Define the tolerance range for the color (adjustable)
    color_tolerance = 40
    color_tolerance2 = 25

    if impactL:
        L_canine = np.array([0, 255 , 0])  # BGR format
        # Define lower and upper bounds for the color with tolerance
        lower_L_Canine = L_canine - color_tolerance
        upper_L_Canine = L_canine + color_tolerance
        mask_L_Canine= cv2.inRange(input_image, lower_L_Canine, upper_L_Canine)
        # Find contours in the masked image
        contours_L_Canine, _ = cv2.findContours(mask_L_Canine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_L_Canine:
            # Get the largest contour
            largest_contour = max(contours_L_Canine, key=cv2.contourArea)
            
            # Fit an ellipse to the largest contour
            ellipse = cv2.fitEllipse(largest_contour)
             # Get the parameters of the fitted ellipse
            center, axes, angle = ellipse
            angle=180-angle
            
            # Calculate the topmost point of the ellipse
            # The topmost point will be at (center_x, center_y - major_axis_length / 2) considering the angle
            center_x, center_y = center
            major_axis_length = max(axes)
            
            # Convert the angle to radians
            angle_rad = np.deg2rad(angle)
            if major_axis_length <80:
                major_axis_length = 80

            # Calculate the topmost point of the ellipse
            topmost_point_x = int(center_x - (major_axis_length / 2) * np.sin(angle_rad))
            topmost_point_y = int(center_y - (major_axis_length / 2) * np.cos(angle_rad))
            L_apex_position = topmost_point_x
            topmost_point = (topmost_point_x, topmost_point_y)            
            cv2.ellipse(input_image, ellipse, (0, 255, 0), 2)
            cv2.circle(input_image, topmost_point, 5, (255, 0, 0), -1)

            
            L_apex_position = topmost_point[0]

            if angle < 90:
                angle_test = angle
            else:
                angle_test = abs(180 - angle)
                
            if angle_test <= 10:
                apexL='Good'
            elif angle_test > 10 and angle_test <= 38:
                apexL='Average'
            elif angle_test > 38:
                apexL= 'Poor'


    if impactR :
        R_first_premolar = np.array([128, 0, 128])  # BGR format
        R_second_premolar = np.array([0, 165, 255])  # BGR format
        R_canine = np.array([255, 0, 255])  # BGR format


        # Define lower and upper bounds for the color with tolerance
        lower_R_Canine = R_canine - color_tolerance
        upper_R_Canine = R_canine + color_tolerance
        mask_R_Canine= cv2.inRange(input_image, lower_R_Canine, upper_R_Canine)
        # Find contours in the masked image
        contours_R_Canine, _ = cv2.findContours(mask_R_Canine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_R_Canine:
            # Get the largest contour
            largest_contour = max(contours_R_Canine, key=cv2.contourArea)
            
            # Fit an ellipse to the largest contour
            ellipse = cv2.fitEllipse(largest_contour)
            
            # Get the parameters of the fitted ellipse
            center, axes, angle = ellipse
            
            # Calculate the topmost point of the ellipse
            # The topmost point will be at (center_x, center_y - major_axis_length / 2) considering the angle
            center_x, center_y = center
            major_axis_length = max(axes)
            
            # Convert the angle to radians
            angle_rad = np.deg2rad(angle)
            if major_axis_length <80:
                major_axis_length = 80
            
            # Calculate the topmost point of the ellipse
            topmost_point_x = int(center_x - (major_axis_length / 2) * np.sin(angle_rad))
            topmost_point_y = int(center_y - (major_axis_length / 2) * np.cos(angle_rad))
            R_apex_position = topmost_point_x
            topmost_point = (topmost_point_x, topmost_point_y)            
            cv2.ellipse(input_image, ellipse, (0, 255, 0), 2)
            cv2.circle(input_image, topmost_point, 5, (255, 0, 0), -1)
            
            R_apex_position = topmost_point[0]

            if angle < 90:
                angle_test = angle
            else:
                angle_test = abs(180 - angle)
                
            if angle_test <= 10:
                apexR='Good'
            elif angle_test > 10 and angle_test <= 33:
                apexR='Average'
            elif angle_test > 33:
                apexR= 'Poor'
    return apexR, apexL


# In[23]:


#Recommandation 
def recomend(impact,angle,apex,vertical,over):
    #All 4 Good
    if angle=="Good" and apex=="Good" and vertical=="Good" and over=="Good":
           print("Straight Forward Impaction")
           return "Straight Forward Impaction"
    #Apex Position Average
    elif angle=="Good" and apex=="Average" and vertical=="Good" and over=="Good":
           print("Mildly Difficult Impaction")
           return "Mildly Difficult Impaction"
    #Apex Position Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Good" and over=="Good":
           print("Mildly Difficult Impaction")
           return "Mildly Difficult Impaction"
    
    
    #overlap Average
    elif angle=="Good" and apex=="Good" and vertical=="Good" and over=="Average":
           print("Mildly Difficult Impaction")
           return "Mildly Difficult Impaction"
    #overlap  and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Good" and over=="Average":
           print("Mildly Difficult Impaction")
           return "Mildly Difficult Impaction"
    #overlap Average and Apex Poor
    elif angle=="Good" and apex=="poor" and vertical=="Good" and over=="Average":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    
    
    #overlap poor
    elif angle=="Good" and apex=="Good" and vertical=="Good" and over=="Poor":
           print("Mildly Difficult Impaction")
           return "Mildly Difficult Impaction"
    #overlap poor and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Good" and over=="Poor":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #overlap poor and Apex Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Good" and over=="Poor":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    
    
    #Vertical Height Average
    elif angle=="Good" and apex=="Good" and vertical=="Average" and over=="Good":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #Vertical Height and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Average" and over=="Good":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #Vertical Height Average and Apex poor 
    elif angle=="Good" and apex=="Poor" and vertical=="Average" and over=="Good":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    
    
    #Vertical Height and Overlap Average
    elif angle=="Good" and apex=="Good" and vertical=="Average" and over=="Averge":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #Vertical Height and Apex and Overlap Average
    elif angle=="Good" and apex=="Average" and vertical=="Average" and over=="Average":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #Vertical Height and Overlap Average and  Apex Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Average" and over=="Average":
           print("Difficult Impaction")
           return "Difficult Impaction"
    
    
    #Vertical Height Average and Overlap Poor
    elif angle=="Good" and apex=="Good" and vertical=="Average" and over=="Poor":
           print("Moderately Difficult Impaction")
           return "Moderately Difficult Impaction"
    #Vertical Height and Apex Average and Overlap Poor
    elif angle=="Good" and apex=="Average" and vertical=="Average" and over=="Poor":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Average and Apex and Overlap Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Average" and over=="Poor":
           print("Difficult Impaction")
           return "Difficult Impaction"
    
    
    #Vertical Height Poor
    elif angle=="Good" and apex=="Good" and vertical=="Poor" and over=="Good":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Poor" and over=="Good":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and Apex Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Poor" and over=="Good":
           print("Difficult Impaction")
           return "Difficult Impaction"
    
    #Vertical Height Poor and overlap average
    elif angle=="Good" and apex=="Good" and vertical=="Poor" and over=="Average":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and overlap average and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Poor" and over=="Average":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and overlap average and Apex Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Poor" and over=="Average":
           print("Difficult Impaction")
           return "Difficult Impaction"
    
    #Vertical Height Poor and overlap poor
    elif angle=="Good" and apex=="Good" and vertical=="Poor" and over=="Poor":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and overlap poor and Apex Average
    elif angle=="Good" and apex=="Average" and vertical=="Poor" and over=="Poor":
           print("Difficult Impaction")
           return "Difficult Impaction"
    #Vertical Height Poor and overlap poor and Apex Poor
    elif angle=="Good" and apex=="Poor" and vertical=="Poor" and over=="Poor":
           print("Difficult Impaction")
           return "Difficult Impaction"
    
    
    #angle average 
    elif angle=="Average" and apex=="Good" and vertical=="Good" and over=="Good":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    #angle average apex average
    elif angle=="Average" and apex=="Average" and vertical=="Good" and over=="Good":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    #angle average and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Good" and over=="Good":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    
    
    #angle average and overlap averge 
    elif angle=="Average" and apex=="Good" and vertical=="Good" and over=="Average":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    ##angle average and overlap averge  apex average
    elif angle=="Average" and apex=="Average" and vertical=="Good" and over=="Average":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    ##angle average and overlap averge  and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Good" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    
    
    #angle average and overlap poor 
    elif angle=="Average" and apex=="Good" and vertical=="Good" and over=="Poor":
        print("Moderately Difficult Impaction")
        return "Moderately Difficult Impaction"
    ##angle average and overlap poor  apex average
    elif angle=="Average" and apex=="Average" and vertical=="Good" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    ##angle average and overlap poor  and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Good" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    
    
    #angle average and vetical average
    elif angle=="Average" and apex=="Good" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical average apex average
    elif angle=="Average" and apex=="Average" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical average and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"


     #angle average and vetical and overlap average
    elif angle=="Average" and apex=="Good" and vertical=="Average" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical and overlap average apex average
    elif angle=="Average" and apex=="Average" and vertical=="Average" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical and overlap average and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Average" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"


      #angle average and vetical average and overlap poor
    elif angle=="Average" and apex=="Good" and vertical=="Average" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical average apex average and overlap poor
    elif angle=="Average" and apex=="Average" and vertical=="Average" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical average and apex poor and overlap poor
    elif angle=="Average" and apex=="Poor" and vertical=="Average" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"


    #angle average and vetical poor
    elif angle=="Average" and apex=="Good" and vertical=="Poor" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical poor apex average
    elif angle=="Average" and apex=="Average" and vertical=="Poor"  and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical poor and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Poor"  and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"


     #angle average and vetical poor and overlap average
    elif angle=="Average" and apex=="Good" and vertical=="Poor" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and veticalpoor and overlap average apex average
    elif angle=="Average" and apex=="Average" and vertical=="Poor" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical poor and overlap average and apex poor
    elif angle=="Average" and apex=="Poor" and vertical=="Poor" and over=="Average":
        print("Complicated Impaction")
        return "Complicated Impaction"


      #angle average and vetical poor and overlap poor
    elif angle=="Average" and apex=="Good" and vertical=="Poor" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle average and vetical poor apex average and overlap poor
    elif angle=="Average" and apex=="Average" and vertical=="Poor" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle average and vetical poor and apex poor and overlap poor
    elif angle=="Average" and apex=="Poor" and vertical=="Poor" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"


    #angle poor 
    elif angle=="Poor" and apex=="Good" and vertical=="Good" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Good" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Good" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"


    #angle poor and overlap average
    elif angle=="Poor" and apex=="Good" and vertical=="Good" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and overlap average and apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Good" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and overlap average and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Good" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"


    #angle poor and overlap poor
    elif angle=="Poor" and apex=="Good" and vertical=="Good" and over=="Poor":
        print("Difficult Impaction")
        return  "Difficult Impaction"
    #angle poor and overlap poor and apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Good" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and overlap poor and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Good" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"


    #angle poor and vertical average
    elif angle=="Poor" and apex=="Good" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and vertical average apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and vertical average and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Average" and over=="Good":
        print("Difficult Impaction")
        return "Difficult Impaction"


    #angle poor and vertical average and over average
    elif angle=="Poor" and apex=="Good" and vertical=="Average" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and vertical average and over average apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Average" and over=="Average":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and vertical average and over average and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Average" and over=="Average":
        print("Complicated Impaction")
        return "Complicated Impaction"


    #angle poor and vertical average and over poor
    elif angle=="Poor" and apex=="Good" and vertical=="Average" and over=="Poor":
        print("Difficult Impaction")
        return "Difficult Impaction"
    #angle poor and vertical average and over poor apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Average" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical average and over poor and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Average" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"


    #angle poor and vertical poor
    elif angle=="Poor" and apex=="Good" and vertical=="Poor" and over=="Good":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical poor and apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Poor" and over=="Good":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical poor and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Poor" and over=="Good":
        print("Complicated Impaction")
        return "Complicated Impaction"


     #angle poor and vertical poor and over average
    elif angle=="Poor" and apex=="Good" and vertical=="Poor" and over=="Average":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical average and over average apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Poor" and over=="Average":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical average and over average and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Poor" and over=="Average":
        print("Complicated Impaction")
        return "Complicated Impaction"


    #angle poor and vertical poor and over poor
    elif angle=="Poor" and apex=="Good" and vertical=="Poor" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical poor and over poor apex average
    elif angle=="Poor" and apex=="Average" and vertical=="Poor" and over=="Poor":
        print("Complicated Impaction")
        return "Complicated Impaction"
    #angle poor and vertical poor and over poor and apex poor
    elif angle=="Poor" and apex=="Poor" and vertical=="Poor" and over=="Poor":
        print("Very Complicated Impaction")
        return "Very Complicated Impaction"

    elif not impact:
        print("No Impaction")
        return "No Impaction"


# In[28]:

def analyze_image(impactL,impactR,angleR,verticalR,overR,apexR,angleL,verticalL,overL,apexL):
    results = {
        "left": {
            "impact": impactL,
            "angulation": angleL,
            "apex_position": apexL,
            "vertical_height": verticalL,
            "overlapping": overL,
            "recommendation": recomend(impactL,angleL,apexL,verticalL,overL)
        },
        "right": {
            "impact": impactR,
            "angulation": angleR,
            "apex_position": apexR,
            "vertical_height": verticalR,
            "overlapping": overR,
            "recommendation": recomend(impactR,angleR,apexR,verticalR,overR)
        }
    }
    return results



# In[ ]:

def main():
    # Load the image
    if len(sys.argv) < 2:
        print("Usage: python script.py <string>")
        return
        
    # Retrieving the string from command-line arguments
    image_path = " ".join(sys.argv[1:])
    
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    impactL,impactR,angleR,verticalR,overR,apexR,angleL,verticalL,overL,apexL = process_image(image_path)

    print("Left Canine: ")
    print("Angulation : {} , Vertical Height : {} , OverLap : {} , Apex Position : {} ".format(angleL,verticalL,overL,apexL))
    recomend(impactL,angleL,apexL,verticalL,overL)

    print("Right Canine : ")
    print("Angulation : {} , Vertical Height : {} , OverLap : {} , Apex Position : {}".format(angleR,verticalR,overR,apexR))
    recomend(impactR,angleR,apexR,verticalR,overR)
    
    results = analyze_image(impactL,impactR,angleR,verticalR,overR,apexR,angleL,verticalL,overL,apexL)
    with open(image_path.replace('.jpg', '.json'), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()

