import cv2
import numpy as np

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)

    red_color = np.array([77, 80, 178])  # BGR format
    faded_color = np.array([139, 139, 192])  # BGR format
    # Define the tolerance range for the color (adjustable)
    color_tolerance = 40
    color_tolerance2 = 25
    # Define lower and upper bounds for the color with tolerance
    lower_red = red_color - color_tolerance
    upper_red = red_color + color_tolerance
    lower_bound = faded_color - color_tolerance2
    upper_bound = faded_color + color_tolerance2
    mask= cv2.inRange(image, lower_red, upper_red)
    mask2= cv2.inRange(image, lower_bound, upper_bound)
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If contours are found
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Fit an ellipse to the contour
        if largest_contour.size > 10:
            
            ellipse = cv2.fitEllipse(largest_contour)
            
            # Extract the angle of inclination from the fitted ellipse
            angle = ellipse[2]
            
            # Display the angle
            print(f"Angle of inclination: {angle} degrees")
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
    if contours2:
        # Get the largest contour
        largest_contour = max(contours2, key=cv2.contourArea)
        # Fit an ellipse to the contour
        if largest_contour.size > 10:
            
            ellipse = cv2.fitEllipse(largest_contour)
            
            # Extract the angle of inclination from the fitted ellipse
            angle = ellipse[2]
            
            angle = 180 - angle
            # Display the angle
            print(f"Angle of inclination: {angle} degrees")
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
            
    if not contours and not contours2:
        print("No red object found in the image.")
    else:
        # Display the image with the detected object
        cv2.imshow('Detected Object', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # save the image
        cv2.imwrite('4_angulation.jpg', image)