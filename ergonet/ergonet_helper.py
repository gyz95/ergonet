import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
import cv2
from .utils import projection,butter_low_pass_filter
from .geometry import matrix_to_axis_angle,axis_angle_to_matrix,matrix_to_rotation_6d,rotation_6d_to_matrix
import smplx
from .render import render, tensor_to_image
import trimesh
from pathlib import Path
from tqdm import tqdm

class ErgoNetHelper:
    def __init__(self, device):
        self.device = device
        self.root_folder_path = str(Path(__file__).resolve().parent)
        self.smpl_path = self.root_folder_path + '/smpl_data/human_models'

    def save_obj_single(self,results,save_path):
        """
        Generate an .obj file based on the predicted results for visualization.

        Args:
            results: outputs from the 'run' function (i.e., the model's predicted results)
            save_path: path and name of the generated .obj file
        """
        if not save_path.endswith('.obj'):
            print('Invalid file type. Mesh can only be saved as .obj file.')
            return

        vertices = results['Vertices']
        gender = results['gender']
        smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                  gender=gender, num_betas=10,
                                  ext='pkl',
                                  batch_size=1)
        scene = trimesh.Scene()
        for idx, vertices_per_mesh in enumerate(vertices):
            mesh = trimesh.Trimesh(vertices_per_mesh, smplx_model.faces, process=False)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            scene.add_geometry(mesh, node_name=f"mesh_{idx}")

        scene.export(save_path)

    def visualize_single(self, original_img, results, save_path, type='vertices'):
        """
        Visualizes ErgoNet's predictions on the original image.

        Args:
            original_img: raw and unprocessed input image.
            results: output from the `run` function (model's predicted results)
            save_path: file path (including name) to save the generated image
            type:
                - vertices: Overlay predicted vertices onto the original image
                - joints: Overlay predicted 2D joints onto the original image

        Returns:
            img: image with an overlay of either predicted vertices or predicted 2D joints
        """
        vertices = results['Vertices']
        j2d_all = results['J2D']
        bboxes = results['BBoxes']
        gender = results['gender']

        camera_intrinsics = results['camera_intrinsics']
        camera_matrix = np.array([[camera_intrinsics[0], 0, camera_intrinsics[2]],
                                  [0, camera_intrinsics[1], camera_intrinsics[3]],
                                  [0, 0, 1]])
        camera_distortion = results['camera_distortion']
        if camera_distortion is not None:
            original_img = cv2.undistort(original_img, camera_matrix, camera_distortion)

        smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                  gender=gender, num_betas=10,
                                  ext='pkl',
                                  batch_size=1).to(self.device)
        img_h, img_w, _ = original_img.shape
        image_shape = [img_w,img_h]
        img = original_img.copy()

        for k in range(len(bboxes)):
            bbox = bboxes[k]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            rel_img_scale = img_h / 720.0
            font_scale = max(0.7, rel_img_scale * 1.0)
            thickness = max(1, int(h / 150.0))
            img = cv2.putText(
                img, str(k), (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
            )

        if type == 'vertices':
            for i in range(len(vertices)):
                rgb_img = img.copy()
                rendered_img = render(vertices[i], smplx_model.faces, rgb_img, camera_intrinsics, image_shape)
                img = tensor_to_image(rendered_img)
                img = img[:, :, ::-1]
        elif type == 'joints':
            for i in range(len(j2d_all)):
                j2d = j2d_all[i]
                for keyp in j2d:
                    x = int(keyp[0])
                    y = int(keyp[1])
                    img = cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        #combined = np.hstack((img, original_img))
        if save_path != None:
            cv2.imwrite(save_path,img)
        return img

    def save_obj_batch(self,results,img_names,save_path):
        """
        Generate an .obj file based on the predicted results for visualization.

        Args:
            results: outputs from the 'run' function (i.e., the model's predicted results)
            img_names: image names corresponding to the provided images.
            save_path: path and name of the generated .obj file
        """
        vertices_all = results['Vertices']
        data_id = results['data_id']
        gender = results['gender']
        smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                   gender=gender, num_betas=10,
                                   ext='pkl',
                                   batch_size=1).to(self.device)
        for i in range(len(img_names)):
            indices_of_img = [j for j, val in enumerate(data_id) if val == i]
            if len(indices_of_img) == 0:
                continue

            vertices = vertices_all[indices_of_img]
            base, _ = os.path.splitext(img_names[i])
            filename = base + ".obj"

            scene = trimesh.Scene()
            for idx, vertices_per_mesh in enumerate(vertices):
                mesh = trimesh.Trimesh(vertices_per_mesh, smplx_model.faces, process=False)
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                scene.add_geometry(mesh, node_name=f"mesh_{idx}")
            save_path_file = save_path + filename
            scene.export(save_path_file)

    def visualize_batch(self,original_images,img_names,results,save_path,type='vertices'):
        """
        Visualizes ErgoNet's predictions on the original image.

        Args:
            original_images: raw and unprocessed input images.
            img_names: image names corresponding to the provided images.
            results: output from the `run` function (model's predicted results)
            save_path: file path (including name) to save the generated image
            type:
                - vertices: Overlay predicted vertices onto the original image
                - joints: Overlay predicted 2D joints onto the original image

        Returns:
            img: image with an overlay of either predicted vertices or predicted 2D joints
        """
        data_id = results['data_id']
        gender = results['gender']
        images_result = []
        smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                   gender=gender, num_betas=10,
                                   ext='pkl',
                                   batch_size=1).to(self.device)
        for i in tqdm(range(len(original_images)), desc="Rendering images", unit="img"):
            original_img = original_images[i]
            img_name = img_names[i]
            indices_of_img = [j for j, val in enumerate(data_id) if val == i]
            if len(indices_of_img) == 0:
                continue

            vertices = results['Vertices'][indices_of_img]
            j2d_all = results['J2D'][indices_of_img]
            bboxes = results['BBoxes'][indices_of_img]

            if len(original_images) != len(results['camera_intrinsics']):
                camera_intrinsics = np.asarray(results['camera_intrinsics'])[indices_of_img][0]
                camera_distortion = np.asarray(results['camera_distortion'])[indices_of_img][0]
            else:
                camera_intrinsics = results['camera_intrinsics'][i]
                camera_distortion = results['camera_distortion'][i]

            camera_matrix = np.array([[camera_intrinsics[0], 0, camera_intrinsics[2]],
                                      [0, camera_intrinsics[1], camera_intrinsics[3]],
                                      [0, 0, 1]])
            if camera_distortion is not None:
                original_img = cv2.undistort(original_img, camera_matrix, camera_distortion)

            img_h, img_w, _ = original_img.shape
            image_shape = [img_w,img_h]
            img = original_img.copy()

            for k in range(len(bboxes)):
                bbox = bboxes[k]
                x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                rel_img_scale = img_h / 720.0
                font_scale = max(0.7, rel_img_scale * 1.0)
                thickness = max(1, int(h / 150.0))
                img = cv2.putText(
                    img, str(k), (x+10, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
                )

            if type == 'vertices':
                for j in range(len(vertices)):
                    rgb_img = img.copy()
                    rendered_img = render(vertices[j], smplx_model.faces, rgb_img, camera_intrinsics, image_shape)
                    img = tensor_to_image(rendered_img)
                    img = img[:, :, ::-1]
            elif type == 'joints':
                for j in range(len(j2d_all)):
                    j2d = j2d_all[j]
                    for keyp in j2d:
                        x = int(keyp[0])
                        y = int(keyp[1])
                        img = cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

            #combined = np.hstack((img, original_img))
            if save_path != None:
                cv2.imwrite(save_path+img_name,img)
            images_result.append(img)
        return images_result

    def k_from_params(self,params):
        """
        Convert camera intrinsics parameters [batch, 4] (fx, fy, cx, cy) to camera intrinsics matrix [batch, 3, 3]
        camera intrinsics matrix: [fx,0,cx,
                                   0,fy,cy,
                                   0,0,1]
        """
        fx, fy, cx, cy = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        batch_size = params.shape[0]
        intrinsics = torch.zeros((batch_size, 3, 3), dtype=params.dtype, device=params.device)

        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1.0

        return intrinsics

    def check_instance(self,x):
        if isinstance(x, torch.Tensor):
            return 'Tensor'
        elif isinstance(x, np.ndarray):
            return 'Array'
        else:
            return None

    def smooth(self,results):
        """
        Apply a low-pass filter to smooth the predicted SMPL parameters and camera translation.
        Intended for continuous predictions of a single person across time.

        Args:
            results: ErgoNet's predicted results for the person of interest

        Returns:
            smoothed_theta (Tensor): Filtered joint rotations.
            unified_beta (Tensor): Constant body shape coefficients across all frames.
            smoothed_cam (Tensor): Filtered camera translations.
        """

        VitPose_J2D = results['VitPose_J2D']
        data_id = results['data_id']
        theta = results['Theta']
        beta = results['Beta']
        bbox = results['BBoxes']
        camera_transl = results['Cam_transl']
        gender = results['gender']
        camera_intrinsics = results['camera_intrinsics']
        camera_distortion = results['camera_distortion']
        num_images = results['num_images']

        length = len(camera_transl)
        if length < 10:
            print('Insufficient valid frames for smoothing')
            return results

        smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                   gender=gender, num_betas=10,
                                   ext='pkl',
                                   batch_size=length).to(self.device)

        if self.check_instance(theta) is None or self.check_instance(beta) is None or self.check_instance(camera_transl) is None:
            print('Wrong Data Format in the Results!')
            return None
        theta = torch.from_numpy(theta).float().to(self.device)
        beta = torch.from_numpy(beta).float().to(self.device)
        camera_transl = torch.from_numpy(camera_transl).float().to(self.device)
        camera_intrinsics_tensor = torch.from_numpy(np.asarray(camera_intrinsics)).float().to(self.device)
        intrinsics_matrix = self.k_from_params(camera_intrinsics_tensor)

        unified_beta = beta.mean(dim=0, keepdim=True).repeat(beta.size(0), 1)

        theta_rotmat = axis_angle_to_matrix(theta)
        theta_6d = matrix_to_rotation_6d(theta_rotmat)
        theta_6d = theta_6d.cpu().numpy()
        theta_6d = np.reshape(theta_6d, (length, 22 * 6))

        smoothed_theta_6d = butter_low_pass_filter(theta_6d)
        smoothed_theta_6d = np.reshape(smoothed_theta_6d, (length, 22, 6))
        smoothed_theta_6d = torch.from_numpy(smoothed_theta_6d).float().to(self.device)
        smoothed_theta_rotmat = rotation_6d_to_matrix(smoothed_theta_6d)
        smoothed_theta_aa = matrix_to_axis_angle(smoothed_theta_rotmat)

        camera_transl = camera_transl.cpu().numpy()
        smoothed_cam = butter_low_pass_filter(camera_transl)
        smoothed_cam = torch.from_numpy(smoothed_cam).float().to(self.device)

        smoothed_theta = smoothed_theta_aa

        with torch.no_grad():
            output = smplx_model(
                betas=unified_beta,
                return_verts=True,
                body_pose=smoothed_theta[:, 1:22, :],
                global_orient=smoothed_theta[:, 0, :],
                transl=smoothed_cam
            )
        smoothed_j3d = output.joints
        smoothed_j2d = projection(smoothed_j3d, intrinsics_matrix)
        smoothed_vertices = output.vertices.cpu().numpy().squeeze()

        smplx2openpose_joints = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                             65]

        smoothed_j3d = smoothed_j3d[:,smplx2openpose_joints,:]
        smoothed_j2d = smoothed_j2d[:,smplx2openpose_joints,:]

        smoothed_cam = smoothed_cam.cpu().numpy().squeeze()
        smoothed_j2d = smoothed_j2d.cpu().numpy().squeeze()
        smoothed_j3d = smoothed_j3d.cpu().numpy().squeeze()
        smoothed_theta = smoothed_theta.cpu().numpy().squeeze()
        unified_beta = unified_beta.cpu().numpy().squeeze()

        smoothed_results = {
            'Cam_transl': smoothed_cam,
            'BBoxes': bbox,
            'J3D': smoothed_j3d,
            'J2D': smoothed_j2d,
            'Theta': smoothed_theta,
            'Beta': unified_beta,
            'Vertices': smoothed_vertices,
            'VitPose_J2D': VitPose_J2D,
            'camera_intrinsics': camera_intrinsics,
            'camera_distortion': camera_distortion,
            'gender': gender,
            'num_images': num_images,
            'data_id': data_id
        }

        return smoothed_results

    def select_poi_single(self,results,poi,poi_type='bbox'):
        """
        Selects the person of interest from detection results.
        Use this only when the poi has not been specified in the `run` function and
        the `run` function outputs results for more than one person.

        Args:
            results: output from the `run` function (model's predicted results)
            poi: data used to identify the person of interest
            poi_type:
                - bbox: [x1, y1, x2, y2] — left-top and right-bottom coordinates of the bounding box
                - aoi: [x1, y1, x2, y2] — left-top and right-bottom coordinates of the area of interest

        Returns:
            results_poi: predicted results corresponding to the selected person of interest
        """
        ergonet_bboxes = results['BBoxes']
        best_idx, best_score = None, -1
        for i in range(len(ergonet_bboxes)):
            ergonet_bbox = ergonet_bboxes[i]
            x, y, w, h = ergonet_bbox[0], ergonet_bbox[1], ergonet_bbox[2], ergonet_bbox[3]
            res_bbox = [x,y,x+w,y+h]
            if poi_type == 'bbox':
                x1, y1, x2, y2 = max(res_bbox[0], poi[0]), max(res_bbox[1], poi[1]), min(res_bbox[2], poi[2]), min(res_bbox[3], poi[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (res_bbox[2] - res_bbox[0]) * (res_bbox[3] - res_bbox[1])
                area2 = (poi[2] - poi[0]) * (poi[3] - poi[1])
                score = inter / (area1 + area2 - inter + 1e-6)
            elif poi_type == 'aoi':  # Intersection area
                x1, y1, x2, y2 = max(res_bbox[0], poi[0]), max(res_bbox[1], poi[1]), min(res_bbox[2], poi[2]), min(res_bbox[3], poi[3])
                score = max(0, x2 - x1) * max(0, y2 - y1)
            else:
                raise ValueError(f"Unknown reference_type: {poi_type}")
            if score > best_score:
                best_score, best_idx = score, i
        if best_score == 0:
            return None

        results['Cam_transl'] = np.expand_dims(results['Cam_transl'][best_idx,:], axis=0)
        results['BBoxes'] = np.expand_dims(results['BBoxes'][best_idx,:], axis=0)
        results['J3D'] = np.expand_dims(results['J3D'][best_idx,:,:], axis=0)
        results['J2D'] = np.expand_dims(results['J2D'][best_idx,:,:], axis=0)
        results['Theta'] = np.expand_dims(results['Theta'][best_idx,:,:], axis=0)
        results['Beta'] = np.expand_dims(results['Beta'][best_idx,:], axis=0)
        results['Vertices'] = np.expand_dims(results['Vertices'][best_idx,:,:], axis=0)

        return results

    def select_poi_batch(self,results,poi,poi_type='bbox'):
        """
        Selects the person of interest from detection results.
        Use this only when the poi has not been specified in the `run` function and
        the `run` function outputs results for more than one person.

        Args:
            results: output from the `run` function (model's predicted results)
            poi: data used to identify the person of interest
            poi_type:
                - bbox: [x1, y1, x2, y2] — left-top and right-bottom coordinates of the bounding box
                - aoi: [x1, y1, x2, y2] — left-top and right-bottom coordinates of the area of interest

        Returns:
            results_poi: predicted results corresponding to the selected person of interest
        """
        original_bboxes = results['BBoxes']
        gender = results['gender']
        num_images = results['num_images']

        if len(poi) != num_images:
            raise ValueError("Dimension of inputs must be the same.")

        data_id = results['data_id']

        cam_transl = []
        j2d = []
        j3d = []
        theta = []
        beta = []
        bboxes = []
        vitpose_j2d = []
        vertices = []
        new_data_id = []
        camera_intrinsics =  []
        camera_distortion =  []

        for i in range(num_images):
            indices_of_img = [j for j, val in enumerate(data_id) if val == i]
            if len(indices_of_img) == 0:
                continue
            image_ergonet_bboxes = original_bboxes[indices_of_img]
            best_idx, best_score = None, -1
            for k in range(len(image_ergonet_bboxes)):
                ergonet_bbox = image_ergonet_bboxes[k]
                x, y, w, h = ergonet_bbox[0], ergonet_bbox[1], ergonet_bbox[2], ergonet_bbox[3]
                res_bbox = [x,y,x+w,y+h]
                if poi_type == 'bbox':
                    x1, y1, x2, y2 = max(res_bbox[0], poi[i][0]), max(res_bbox[1], poi[i][1]), min(res_bbox[2], poi[i][2]), min(res_bbox[3], poi[i][3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (res_bbox[2] - res_bbox[0]) * (res_bbox[3] - res_bbox[1])
                    area2 = (poi[i][2] - poi[i][0]) * (poi[i][3] - poi[i][1])
                    score = inter / (area1 + area2 - inter + 1e-6)
                elif poi_type == 'aoi':  # Intersection area
                    x1, y1, x2, y2 = max(res_bbox[0], poi[i][0]), max(res_bbox[1], poi[i][1]), min(res_bbox[2], poi[i][2]), min(res_bbox[3], poi[i][3])
                    score = max(0, x2 - x1) * max(0, y2 - y1)
                else:
                    raise ValueError(f"Unknown reference_type: {poi_type}")
                if score > best_score:
                    best_score, best_idx = score, k
            if best_score == 0:
                continue

            cam_transl.append(results['Cam_transl'][indices_of_img][best_idx,:])
            j2d.append(results['J2D'][indices_of_img][best_idx,:,:])
            j3d.append(results['J3D'][indices_of_img][best_idx,:,:])
            theta.append(results['Theta'][indices_of_img][best_idx,:,:])
            beta.append(results['Beta'][indices_of_img][best_idx,:])
            bboxes.append(results['BBoxes'][indices_of_img][best_idx,:])
            vertices.append(results['Vertices'][indices_of_img][best_idx,:,:])
            vitpose_j2d.append(results['VitPose_J2D'][indices_of_img][best_idx,:,:])
            camera_intrinsics.append(results['camera_intrinsics'][i])
            camera_distortion.append(results['camera_distortion'][i])
            new_data_id.append(i)

        if len(cam_transl) == 0:
            return None

        cam_transl = np.asarray(cam_transl)
        j2d = np.asarray(j2d)
        j3d = np.asarray(j3d)
        theta = np.asarray(theta)
        beta = np.asarray(beta)
        bboxes = np.asarray(bboxes)
        vertices = np.asarray(vertices)
        vitpose_j2d = np.asarray(vitpose_j2d)

        if len(cam_transl) == 1:
            cam_transl = np.expand_dims(cam_transl, axis=0)
            j3d = np.expand_dims(j3d, axis=0)
            j2d = np.expand_dims(j2d, axis=0)
            theta = np.expand_dims(theta, axis=0)
            beta = np.expand_dims(beta, axis=0)
            vertices = np.expand_dims(vertices, axis=0)
            bboxes = np.expand_dims(bboxes, axis=0)
            vitpose_j2d = np.expand_dims(vitpose_j2d, axis=0)


        results = {
            'Cam_transl': cam_transl,
            'BBoxes': bboxes,
            'J3D': j3d,
            'J2D': j2d,
            'Theta': theta,
            'Beta': beta,
            'Vertices': vertices,
            'VitPose_J2D': vitpose_j2d,
            'camera_intrinsics': camera_intrinsics,
            'camera_distortion': camera_distortion,
            'gender': gender,
            'num_images': num_images,
            'data_id': new_data_id
        }
        return results

    def make_video(self, images, save_path, fps=30):
        first = images[0]
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        for img in images:
            frame = img
            writer.write(frame)

        writer.release()
