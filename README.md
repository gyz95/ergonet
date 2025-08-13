# ErgoNet: Predicting Human Body Pose and Shape from Monocular Images for Ergonomics Assessments

<img width="2844" height="1038" alt="demo_image" src="https://github.com/user-attachments/assets/e0ff7296-eadf-4e5e-9813-cb801dfd8f32" />

## Installation
```bash
git clone ssh://git.amazon.com/pkg/ErgoNetPredicter
cd ErgoNetPredicter
pip install -e.
```

## Features 
### Input Processing and Detection:
- Process both static images and video sequences as input data
- Perform robust multi-person detection using state-of-the-art object detection algorithms to generate precise bounding boxes around each individual

### Pose Estimation:
- Predict 3D joint positions for each detected person in the camera's coordinate system
- Predict 2D joint position for each detected person in the image
- Predict 3D joint rotations representing in the axis-angle format for comprehensive pose understanding

### Anthropometric Analysis using Monocular T-Pose/A-Pose Videos:
- Reconstruct detailed 3D body shapes for ergonomic assessment
- Compute key anthropometric measurements critical for ergonomic evaluation

### Post-Processing:
- Identify the person of interest by defining an area of interest
- Use a low-pass filter to smooth body motions in video inputs
- Visualize predictions using 3D mesh renderings

## Key Components    
### - ErgoNetSingleProcessor: 
**_from ergonet import ErgoNetSingleProcessor_**

It processes a single image and predicts the poses of each person in the image. To use the single processor, simply provide an image read by OpenCV, specify the computing device (cpu or cuda or mps), and set the gender (neutral, male, or female). Camera intrinsic parameters and distortion coefficients are optional and can be set to None. However, they are required if you want the predictions to be in real-world metric scale—that is, to obtain the actual 3D positions of each body joint.

Camera Intrinsics Parameters 

[fx, fy, cx, cy]
- fx - focal length in x-axis (in pixel)
- fy - focal length in y-axis (in pixel)
- cx - principal point in x-axis (in pixel)
- cy - principal point in y-axis (in pixel)

Camera Distortion Coefficients 

[k1, k2, p1, p2, k3]
- k1, k2, k3 - radial distortion coefficients
- p1, p2 - tangential distortion coefficients

### - ErgoNetBatchProcessor:
**_from ergonet import ErgoNetBatchProcessor_**

It processes multiple image and predicts the poses of each person in each image. To use the batch processor, simply provide a list of images read by OpenCV, specify the computing device (cpu or cuda, or mps), and set the gender (neutral, male, or female). Note that the selected gender will be applied uniformly to all detected individuals across all images.

### - ErgoNetHelper:
**_from ergonet import ErgoNetHelper_**

It takes ErgoNet's predicted results as input and provides multiple post-processing services, including identifying the person of interest, filtering the predicted body motions, and visualizing the results.

## Example Usages  
### Static Images:
```bash
from ergonet import ErgoNetSingleProcessor, ErgoNetHelper
import cv2

device = 'cpu' 
gender = 'neutral'
camera_intrinsics = None  
camera_distortion = None  

image_path = '.../demo_image.jpg'
save_path_img = '.../demo_image_results.jpg'
image = cv2.imread(image_path)

ErgoNet = ErgoNetSingleProcessor(device)
ErgoNet_Helper = ErgoNetHelper(device)
results = ErgoNet.run(image, camera_intrinsics=camera_intrinsics, camera_distortion=camera_distortion, gender=gender)
vis_img = ErgoNet_Helper.visualize_single(image, results, save_path_img, type='vertices')
```
### Short Videos (Recommended Duration < 5 Seconds or < 300 Frames):
```bash
from ergonet import ErgoNetBatchProcessor, ErgoNetHelper
import cv2

device = 'cuda'
gender = 'female' 
video_path = '.../demo_video.MP4'
video_results_path = '.../demo_output.MP4'
images = []
img_names = []
camera_intrinsics = []
camera_distortion = []
pois = [] ##[x1, y1, x2, y2] bounding boxes used to identify the person of interest (poi)

video_cap = cv2.VideoCapture(video_path)
id = 0
while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    if id == 0: 
        roi = cv2.selectROI("Select ROI", frame.copy())
        roi_x, roi_y, roi_w, roi_h = roi

    pois.append([roi_x,roi_y,roi_x+roi_w,roi_y+roi_h])
    images.append(frame)
    camera_intrinsics.append(None)
    camera_distortion.append(None)
    img_names.append(str(id)+'.jpg')

    cv2.imshow("Frame", frame)
    id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_cap.release()
cv2.destroyAllWindows()

ErgoNet_Batch = ErgoNetBatchProcessor(device)
ErgoNet_Helper = ErgoNetHelper(device)

results_multi_person = ErgoNet_Batch.run(images,img_names,camera_intrinsics,camera_distortion,gender=gender,joint_optimize=True)
results_poi = ErgoNet_Helper.select_poi_batch(results_multi_person,poi=pois)
smoothed_results_poi = ErgoNet_Helper.smooth(results_poi)
vis_imgs = ErgoNet_Helper.visualize_batch(images,img_names,smoothed_results_poi,save_path=None,type='vertices') 
ErgoNet_Helper.make_video(vis_imgs,save_path=video_results_path,fps=60)
```
### Critical Outputs: 

The .run function always returns a dictionary containing the following critical keys:
- Cam_transl: 3D camera translation (Num_People × 3) - the camera’s position in 3D space relative to each person’s pelvis
- BBoxes: Detected bounding boxes (Num_People × 4) - in the format of (x, y, w, h)
- J3D: Predicted 3D joint positions (Num_People x 25 x 3) - COCO 25/OpenPose 25 Joints
- J2D: Predicted 3D joint positions (Num_People x 25 x 2) - COCO 25/OpenPose 25 Joints
- Theta: Predicted 3D joint rotations (Num_People x 22 x 3) - SMPLX Joints 

### Skeletons and Joint Orders:
Openpose 25/ COCO 25 Joints:

0 - Nose, 1 - Neck, 2 - Right shoulder, 3 - Right elbow, 4 - Right wrist, 5 - Left shoulder, 6 - Left elbow, 7 - Left wrist, 8 - Mid hip, 9 - Right hip, 10 - Right knee, 11 - Right ankle, 12 - Left hip, 13 - Left knee, 14 - Left ankle, 15 - Right eye, 16 - Left eye, 17 - Right ear, 18 - Left ear, 19 - Left big-toe, 20 - Left small-toe, 21 - Left heel, 22 - Right big-toe, 23 - Right small-toe, 24 - Right heel

SMPLX Joints:

0 - Pelvis, 1 - Left hip, 2 - Right hip, 3 - Spine1, 4 - Left knee, 5 - Right knee, 6 - Spine2, 7 - Left ankle, 8 - Right ankle, 9 - Spine3, 10 - Left foot, 11 - Right foot, 12 - Neck, 13 - Left collar, 14 - Right collar, 15 - Head, 16 - Left shoulder, 17 - Right shoulder, 18 - Left elbow, 19 - Right elbow, 20 - Left wrist, 21 - Right wrist




