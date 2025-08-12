from ergonet import ErgoNetSingleProcessor, ErgoNetHelper
import cv2

device = 'mps' #['cuda','cpu','mps']
gender = 'neutral' #['neutral','male','female']
camera_intrinsics = None  # [fx,fy,cx,cy]
camera_distortion = None  # [k1, k2, p1, p2, k3]

image_path = '/Users/guoyang/Desktop/ErgoNet_pkg/tests/test_images/test65.jpg'
save_path_img = '/Users/guoyang/Desktop/development/ergonet/test/test65_render.jpg'
save_path_obj = '/Users/guoyang/Desktop/development/ergonet/test/test65_mesh.obj'
save_path_results = '/Users/guoyang/Desktop/development/ergonet/test/test65_result.npy'
image = cv2.imread(image_path)

ErgoNet = ErgoNetSingleProcessor(device)
ErgoNet_Helper = ErgoNetHelper(device)
results = ErgoNet.run(image, camera_intrinsics=camera_intrinsics, camera_distortion=camera_distortion, gender=gender)
vis_img = ErgoNet_Helper.visualize_single(image, results, save_path_img, type='vertices')  # ['vertices','joints']
ErgoNet_Helper.save_obj_single(results, save_path_obj)
