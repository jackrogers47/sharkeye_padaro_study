import cv2
import os
import argparse


from ultralytics import YOLO

from video_processing import ar_resize
from tracker_logic import Track, SharkTracker
from output import output, draw_bounding_box


#helper function for timestamping - in progress not working properly
def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return str(minutes) + ':' + str(round(seconds)) 


# gpu = True if Apple mps is available
# imgsz = desired pixel width for inference
# video_dir = directory of videos to run inference on
# altitude = set target altitude of transect, default = 30
def run_inference(gpu=False, imgsz=720, video_dir='survey_video', altitude=30):

    #load the model with the weights located at weights path
    weights_path = 'model_weights/exp1v8sbest.pt'
    model = YOLO(weights_path)

    #assign desired frame rate for inference based on the availability of Apple mps gpu, we will sample our 
    # survey videos at the desired frame rate before running inference on them
    if gpu:
        desired_frame_rate = 8
    elif not gpu:
        desired_frame_rate = 4

    # make function to concatenate videos if needed here

    #list of full file paths to videos for each video in video_dir which in default is survey_video
    videos = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if not file.startswith('.')]

    #initiate list to save all high confidence sharks 
    final_shark_list = []
    #initiate list to save all low confidence tracked objects
    final_low_conf_tracks_list = []

    #iterate through each video in the videos list
    for video in videos:
        cap = cv2.VideoCapture(video)

        original_frame_width = cap.get(3)
        original_frame_height = cap.get(4)

        # given the original frame width and height and desired pixel width, return the desired [pixel_height, pixel_width] 
        # to resize images to before running inference that maintains the original aspect ratio
        resize_img_w_and_h = ar_resize(original_frame_width, original_frame_height, imgsz)
        
        # get original frame rate
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # get rate at which to sample survey_video
        frame_sample_rate = round(original_fps/desired_frame_rate)

        #initiate tracker object 
        st = SharkTracker(altitude, desired_frame_rate)

        frame_no = 0

        #iterate over each frame in video
        while cap.isOpened():
            success = cap.grab()

            #reducing video frame rate here
            if success and frame_no % frame_sample_rate == 0:
                _, frame = cap.retrieve()

                # running inference here, results[0] has attributes boxes, id, conf, which are lists 
                # containing the ids, bounding boxes, and confidence  information about each detection in frame

                # note: frame rate should relate to how we set iou
                # conf = min confidence to make a detection
                if gpu:
                    results = model.track(frame, conf=.41, device='mps', imgsz=resize_img_w_and_h, iou=0.39, show=True, verbose=False, persist=True)
                elif not gpu:
                    results = model.track(frame, conf=.41, imgsz=resize_img_w_and_h, iou=0.39, show=True, verbose=False, persist=True)
                
                # Get the boxes ,classes and track IDs of frame
                boxes = results[0].boxes.xywh.cpu().tolist()
                confidence = results[0].boxes.conf.cpu().tolist() 
                track_ids = results[0].boxes.id
                # handles a yolo empty list error
                if track_ids == None:
                    track_ids = []
                else:
                    track_ids= track_ids.cpu().tolist()
                    
                timestamp = 'na'

                detections_list = zip(track_ids, boxes, confidence)

                #update tracker with detections from frame, returns an appended list of all tracked items
                all_tracks = st.update_tracker(detections_list, frame, original_frame_width, timestamp)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # exits the loop if the video is over
            elif not success:
                break

            frame_no += 1
        


        for trk in all_tracks:
            if trk.confirmed:
                final_shark_list.append(trk)
            else:
                final_low_conf_tracks_list.append(trk)


    # save ann info
    output(final_shark_list, final_low_conf_tracks_list)


    #return final shark list


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action =argparse.BooleanOptionalAction, help='True or False this is a Macbook with an m1, m2, or m3 chip')
    parser.add_argument('--imgsz', type=int, default=720, help='image height for inference (pixels)')
    parser.add_argument('--video_dir', type=str, default='survey_video', help='folder where videos to process exist')
    parser.add_argument('--altitude', type=int, default=30, help='survey flight altitude (meters)')
    opt = parser.parse_args()
    return opt

def main(opt):
    run_inference(**vars(opt))


if __name__=='__main__':
    opt = parse_opt()    
    main(opt)



