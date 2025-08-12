from ergonet import ErgoNetBatchProcessor
from ergonet import ErgoNetHelper
import cv2

device = 'mps' # ['cuda','cpu','mps'] warning: using cpu to run the batch processor can be very time-consuming
gender = 'female' # ['neutral','male','female']

video_path = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/S31_Cam2_Lift_Lower_25_6614.024958298706.MP4'
save_path_images = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/result_images/'
save_path_obj = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/result_obj/'
images = []
img_names = []
camera_intrinsics = []
camera_distortion = []
pois = []

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

    #cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_cap.release()
cv2.destroyAllWindows()

print(img_names)

ErgoNet_Batch = ErgoNetBatchProcessor(device)
ErgoNet_Helper = ErgoNetHelper(device)
results = ErgoNet_Batch.run(images,img_names,camera_intrinsics,camera_distortion,gender=gender,joint_optimize=True)
poi_results = ErgoNet_Helper.select_poi_batch(results,poi=pois,poi_type='aoi')
smoothed_results = ErgoNet_Helper.smooth(poi_results)
vis_imgs = ErgoNet_Helper.visualize_batch(images,img_names,smoothed_results,save_path=None,type='vertices') #['vertices','joints']
ErgoNet_Helper.make_video(vis_imgs,save_path='/Users/guoyang/Desktop/ErgoNet_pkg/tests/result_images/out.mp4',fps=60)
