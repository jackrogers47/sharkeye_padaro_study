This repository contain the logic to run inference on juvenile Great White Shark survey flights at Padaro Beach. 

run_inference.py contains unique logic for the sharkeye project. Unique logic includes our human in the loop approach where tracked objects were sorted into high (~0.70) confidence and low (~0.40) confidence detections. Researchers then went through and approved most high confidence detections and threw out most low confidence detections in expedited post processing. Bounding box measuring and drone altitude were used to calculate shark size.

Also note, this repository was forked from the YOLOv8 repository as at the time of development a critical script for tracking had a bug in the version currently published by Ultralitics but the bug was fixed in the YOLOv8 github repo. Thus, all of the YOLOv8 backend exists in this repo as well.

more about SharkEye can be learned at: https://sharkeye.org/
