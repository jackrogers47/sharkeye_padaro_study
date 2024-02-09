import os
import pandas as pd
import cv2

from tracker_logic import Track, SharkTracker



def convert_bbox_center_to_corners(x_center, y_center, width, height):
    # Calculate half-width and half-height
    half_width = width / 2
    half_height = height / 2

    # Calculate top-left and bottom-right coordinates
    top_left_x = int(x_center - half_width)
    top_left_y = int(y_center - half_height)
    bottom_right_x = int(x_center + half_width)
    bottom_right_y = int(y_center + half_height)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def draw_max_conf_bounding_box(image, bbox, object_id):
    x, y, width, height = bbox

    # Convert float coordinates to integers
    x, y, width, height = int(x), int(y), int(width), int(height)

    xtl, ytl, xbr, ybr = convert_bbox_center_to_corners(x, y, width, height)

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, thickness)

    # Display object information
    text = f"Object ID: {object_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (xbr - 30, ybr + 30)
    cv2.putText(image, text, text_position, font, font_scale, color, font_thickness)
    return image 

'''
def draw_bounding_box(image, bbox, object_id, size, measured_on):
    x, y, width, height = bbox

    # Convert float coordinates to integers
    x, y, width, height = int(x), int(y), int(width), int(height)

    xtl, ytl, xbr, ybr = convert_bbox_center_to_corners(x, y, width, height)

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, thickness)

    # Display object information
    text = f"Object ID: {object_id}, Size (ft): {size:.2f}, Measured on: {measured_on}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (xbr - 30, ybr + 30)
    cv2.putText(image, text, text_position, font, font_scale, color, font_thickness)
    return image 
'''


def save_image_with_bbox(image, filename, path):
    filepath = path + '/' + filename + '.jpeg'
    cv2.imwrite(filepath, image)



def output(final_shark_list):
    if not os.path.exists(os.path.join(os.getcwd(),'results')):
        os.makedirs('results')
    
    #TODO make this more robust

    if len([f for f in os.listdir('results') if not f.startswith('.')]) == 0:
        survey_number = 0
    else: 
        max = 0
        for fname in [f for f in os.listdir('results') if not f.startswith('.')]:
            survey_num = int(fname[6:])
            if survey_num > max:
                max = survey_num
            survey_number = max + 1
    

    survey_filename = 'survey' + str(survey_number)
    survey_path = os.path.join('results', survey_filename)

    images_for_verification_path = os.path.join(survey_path, 'images_for_manual_verification')
    os.makedirs(images_for_verification_path)

    
    shark_df = pd.DataFrame(columns=['Object_ID', 'Date', 'Video_Number', 'Timestamp', 'Size_(ft)', 'Measured_On',  'Confidence_Category'])
    for s in final_shark_list:
        if s.confirmed:
            conf_category = 'high'
        else:
            conf_category = 'low'
        row_df = pd.DataFrame([{'Object_ID': s.id, 
                                'Date': s.date, 
                                'Video_Number': s.video_number,
                                'Timestamp': s.timestamp,
                                'Size_(ft)': s.size, 
                                'Measured_On': s.measured_on,  
                                'Confidence_Category': conf_category}])
        
        #TODO: fix, in future versions of pandas, we will not be able to concatenate to empty df
        shark_df = pd.concat([shark_df, row_df], ignore_index=True)

        frame_filename = 'track_' + str(round(s.id)).rjust(5, '0')
        save_image_with_bbox(draw_max_conf_bounding_box(s.max_conf_frame, s.box, s.id), frame_filename, images_for_verification_path)
    
    shark_df.to_csv(os.path.join(survey_path, 'inference_results_table.csv'))



