from ergonet import ErgoNetBatchProcessor
from ergonet import ErgoNetHelper
import cv2
import os
import re

device = 'mps' # ['cuda','cpu','mps'] warning: using cpu to run the batch processor can be very time-consuming
gender = 'neutral' # ['neutral','male','female']
images_path = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/test_images/' # Please specify the directory where the raw images are located
save_path_images = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/result_images/' # Please specify the directory where you would like to save the rendered images
save_path_obj = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/result_obj/' # Please specify the directory where you would like to save the object files
image_names = os.listdir(images_path)
image_names = [f for f in image_names if f != '.DS_Store']
image_names.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
print(image_names)

images = []
img_names = []
camera_intrinsics = []
camera_distortion = []
pois = []
for image_name in image_names:
    image_path = os.path.join(images_path,image_name)
    read_image = cv2.imread(image_path)
    if read_image is None:
        continue
    images.append(read_image)
    img_names.append(image_name)
    camera_intrinsics.append(None)
    camera_distortion.append(None)

ErgoNet_Batch = ErgoNetBatchProcessor(device)
ErgoNet_Helper = ErgoNetHelper(device)
results = ErgoNet_Batch.run(images,img_names,camera_intrinsics,camera_distortion,gender=gender)
vis_imgs = ErgoNet_Helper.visualize_batch(images,img_names,results,save_path_images,type='vertices')
ErgoNet_Helper.save_obj_batch(results,img_names,save_path_obj)