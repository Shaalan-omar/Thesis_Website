
import cv2
import numpy as np
import os.path
from roboflow import Roboflow
import matplotlib.pyplot as plt
import os
import sys



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



def main ():

    #preprocessing 
    if len(sys.argv) < 2:
        print("Usage: python script.py <string>")
        return
        
    # Retrieving the string from command-line arguments
    image_path = " ".join(sys.argv[1:])
    
    preproc(image_path)


    #yolo
    image_name = os.path.basename(image_path)
    image_name = image_name.split('.')[0]
    original_image=yolo( image_name)
    output_folder = "static/processed"
    output_path = os.path.join(output_folder, f'{image_name}.jpg')
    cv2.imwrite(output_path, original_image)
    
    
    
if __name__ == "__main__":
    main()
