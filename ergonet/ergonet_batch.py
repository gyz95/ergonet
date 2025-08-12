import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
import cv2
from .common.imutils import process_image
from .predictor import ErgoNetPredictor
from .utils import projection,cam_crop2full_raw
from .geometry import matrix_to_axis_angle,axis_angle_to_matrix,matrix_to_euler_angles
import smplx
from tqdm import tqdm
from .easy_ViTPose.inference import VitInference
from rfdetr import RFDETRBase,RFDETRLarge
from pathlib import Path

class ErgoNetBatchProcessor:
    def __init__(self, device):
        #self.root_folder_path = os.getcwd()
        self.root_folder_path = str(Path(__file__).resolve().parent)
        smpl_path = self.root_folder_path + '/smpl_data/human_models'
        ckpt_path = self.root_folder_path + '/checkpoints/model_ckpt_385000.pth'
        smpl_mean_path = self.root_folder_path + '/smpl_data/smpl_mean_params.npz'
        vitpose_path = self.root_folder_path + '/checkpoints/vitpose-h-wholebody.pth'

        self.ckpt_path = ckpt_path
        self.smpl_path = smpl_path
        self.smpl_mean_path = smpl_mean_path
        self.device = device

        self.predictor = self.get_model(ckpt_path, smpl_mean_path)
        self.predictor.eval()

        self.PoseEstimator2D = VitInference(vitpose_path, model_name='h', is_video=False, device=device)

    def update_model_weights(self, new_model_path):
        """
        Update the weights of the model.

        Args:
            model_path: path to the model's weights (.pth file)
        """
        self.ckpt_path = new_model_path
        self.predictor = self.get_model(new_model_path, self.smpl_mean_path)
        self.predictor.eval()

    def penalty_coefficient(self, x, x0, x1, y0, y1, beta=2.0):
        """
        Computes penalty coefficients based on joint confidences.

        Args:
            x: joint confidences (Batch × Number of Joints)
            x0: lower bound of the confidence range (0 ≤ x0 ≤ 1)
            x1: upper bound of the confidence range (0 ≤ x1 ≤ 1, x1 > x0)
            y0: penalty value at x0 (0 ≤ y0 ≤ 1)
            y1: penalty value at x1 (0 ≤ y1 ≤ 1, y1 < y0)
            beta: controls the rate of change; higher values cause the penalty
                  to drop more rapidly as confidence increases

        Example:
            With x0=0.3, x1=0.7, y0=0.7, y1=0.1:
                - penalty is 0.7 when confidence = 0.3
                - penalty decreases to 0.1 when confidence = 0.7
            In other words, higher confidences yield lower penalties
        """
        t = ((x - x0) / (x1 - x0)).clamp(0, 1)
        s = (t * t * (3 - 2 * t)) ** beta
        return (1 - s) * y0 + s * y1

    def optimize_joint(self, theta, beta, cam_full, camera_matrix,
                       smplx_model, body_j2d, body_j2d_conf, hand_kp2d, hand_kp2d_conf, num_iter=1000):
        """
        Refines ErgoNet's predicted results using ViTPose's joint predictions.

        Args:
            theta: original pose parameters (theta)
            beta: original shape parameters (beta)
            cam_full: original camera translation
            camera_matrix: camera intrinsic parameters in matrix form
            smplx_model: pre-initialized SMPL-X model
            body_j2d: 2D body joint positions predicted by ViTPose
            body_j2d_conf: confidence scores for body joint predictions
            hand_kp2d: 2D hand joint positions predicted by ViTPose
            hand_kp2d_conf: confidence scores for hand joint predictions
            num_iter: number of optimization iterations

        Returns:
            optimized_theta: refined pose parameters
            beta_copy: copy of the original beta (unchanged)
            optimized_cam: refined camera translation
        """
        body_j2d_conf = torch.clamp(body_j2d_conf, 0.0, 1.0)
        hand_kp2d_conf = torch.clamp(hand_kp2d_conf, 0.0, 1.0)
        smplx2smpl_joints = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                             65]
        smpl_joint_index = [0, 16, 15, 18, 17, 1, 5, 2, 6, 3, 7, 4, 12, 9, 8, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24]
        smplx_hand_joints = [37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69,
                             31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73,
                             49, 50, 51, 74, 46, 47, 48, 75]

        delete_index_smpl = [5, 12, 13, 14]
        delete_index_vit = [11, 12]
        smpl_joint_index = [val for i, val in enumerate(smpl_joint_index) if i not in delete_index_smpl]
        vit_index = [i for i in range(23) if i not in delete_index_vit]

        body_j2d = body_j2d[:, vit_index, :]
        body_j2d_conf = body_j2d_conf[:, vit_index]

        theta_index = [4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
        theta_conf = torch.zeros_like(body_j2d_conf[:, 0:11])
        theta_conf[:, 0] = body_j2d_conf[:, 11]
        theta_conf[:, 1] = body_j2d_conf[:, 12]
        theta_conf[:, 2] = body_j2d_conf[:, [13, 15, 16, 17]].mean(dim=1)
        theta_conf[:, 3] = body_j2d_conf[:, [14, 18, 19, 20]].mean(dim=1)
        theta_conf[:, 4] = body_j2d_conf[:, 0:5].mean(dim=1)
        theta_conf[:, 5] = body_j2d_conf[:, 5]
        theta_conf[:, 6] = body_j2d_conf[:, 6]
        theta_conf[:, 7] = body_j2d_conf[:, 7]
        theta_conf[:, 8] = body_j2d_conf[:, 8]
        theta_conf[:, 9] = hand_kp2d_conf[:, 0:21].mean(dim=1)
        theta_conf[:, 10] = hand_kp2d_conf[:, 21:42].mean(dim=1)

        beta_copy = beta.clone()
        beta_copy.requires_grad = False

        global_rot = theta[:, 0, :].clone().requires_grad_(True)
        selected_joint_rot = theta[:, theta_index, :].clone().requires_grad_(True)

        initial_global_rot = global_rot.clone().detach()
        initial_selected_joint_rot = selected_joint_rot.clone().detach()

        lower = cam_full - 0.4
        upper = cam_full + 0.4
        cam_full_raw = torch.nn.Parameter(torch.zeros_like(cam_full))

        optimizer = torch.optim.AdamW([global_rot, selected_joint_rot, cam_full_raw], lr=1e-2)

        max_angle_rad = torch.tensor(180.0 * np.pi / 180.0, device=selected_joint_rot.device)

        with tqdm(range(num_iter)) as pbar:
            for i in pbar:
                optimized_cam = torch.sigmoid(cam_full_raw) * (upper - lower) + lower

                theta_tmp = theta.clone()
                angle_magnitudes = torch.norm(selected_joint_rot, dim=-1, keepdim=True)
                scaling_factors = torch.clamp(max_angle_rad / (angle_magnitudes + 1e-8), max=1.0)
                selected_joint_rot_clipped = selected_joint_rot * scaling_factors
                theta_tmp[:, theta_index, :] = selected_joint_rot_clipped

                output = smplx_model(
                    betas=beta_copy,
                    return_verts=False,
                    body_pose=theta_tmp[:, 1:, :],
                    global_orient=global_rot,
                    transl=optimized_cam
                )

                pred_j3d = output.joints
                pred_j2d = projection(pred_j3d, camera_matrix)
                pred_hand_j2d = pred_j2d[:, smplx_hand_joints, :]
                pred_j2d = pred_j2d[:, smplx2smpl_joints, :]
                pred_j2d = pred_j2d[:, smpl_joint_index, :]

                joints_2d_diff = (pred_j2d - body_j2d).abs().mean(dim=-1)
                j2d_loss = (body_j2d_conf ** 2) * joints_2d_diff
                j2d_loss = j2d_loss.mean(dim=-1)

                hand_joints_2d_diff = (pred_hand_j2d - hand_kp2d).abs().mean(dim=-1)
                hand_j2d_loss = (hand_kp2d_conf ** 2) * hand_joints_2d_diff
                hand_j2d_loss = hand_j2d_loss.mean(dim=-1)

                rot_diff = global_rot - initial_global_rot
                angle_diff_rad = torch.norm(rot_diff, dim=-1)
                angle_diff_deg = angle_diff_rad * (180.0 / torch.pi)
                rot_reg_loss = torch.clamp(angle_diff_deg - 10.0, min=0.0).mean()

                # --- Confidence-adaptive joint change penalty ---
                theta_delta = selected_joint_rot - initial_selected_joint_rot
                delta_norm = torch.norm(theta_delta, dim=-1)  # [B, 11]

                beta1 = 0.2
                beta2 = 0.2
                beta3 = 0.2

                seg1 = self.penalty_coefficient(theta_conf, 0.1, 0.4, 1.0, 0.05, beta=beta1)
                seg2 = self.penalty_coefficient(theta_conf, 0.4, 0.9, 0.05, 0.001, beta=beta2)
                seg3 = self.penalty_coefficient(theta_conf, 0.9, 1.0, 0.001, 0.0, beta=beta3)

                delta_weight = torch.where(
                    theta_conf < 0.1, torch.ones_like(theta_conf),
                    torch.where(
                        theta_conf < 0.4, seg1,
                        torch.where(theta_conf < 0.9, seg2, seg3)
                    )
                )

                delta_penalty = (delta_weight * (delta_norm ** 2)).mean()

                # --- Confidence-adaptive biomechanical limit penalty ---
                RAD2DEG = 180.0 / torch.pi
                B = selected_joint_rot.shape[0]
                R = axis_angle_to_matrix(selected_joint_rot.reshape(-1, 3))  # [B*11,3,3]
                eul_deg = (matrix_to_euler_angles(R, "XYZ") * RAD2DEG).reshape(B, 11, 3)  # [B,11,3]

                min_deg = torch.full((1, 11, 3), float("-inf"), device=selected_joint_rot.device,
                                     dtype=selected_joint_rot.dtype)
                max_deg = torch.full((1, 11, 3), float("+inf"), device=selected_joint_rot.device,
                                     dtype=selected_joint_rot.dtype)
                joint_limit_config = {
                    2: [(2, -20.0, 20.0), (0, -20.0, 40.0)],
                    3: [(2, -20.0, 20.0), (0, -20.0, 40.0)],
                    9: [(2, -60.0, 60.0), (1, -20.0, 30.0)],
                    10: [(2, -60.0, 60.0), (1, -30.0, 20.0)],
                }
                for j_idx, axes in joint_limit_config.items():
                    for axis_idx, lo, hi in axes:
                        min_deg[0, j_idx, axis_idx] = lo
                        max_deg[0, j_idx, axis_idx] = hi

                dist_upper = (eul_deg - max_deg).clamp_min(0)  # [B,11,3]
                dist_lower = (min_deg - eul_deg).clamp_min(0)  # [B,11,3]
                dist = dist_upper + dist_lower  # [B,11,3]

                beta1 = beta2 = beta3 = 0.2
                seg1 = self.penalty_coefficient(theta_conf, 0.1, 0.7, 1.0, 0.05, beta=beta1)  # [B,11]
                seg2 = self.penalty_coefficient(theta_conf, 0.7, 0.9, 0.05, 0.001, beta=beta2)
                seg3 = self.penalty_coefficient(theta_conf, 0.9, 1.0, 0.001, 0.0, beta=beta3)
                penalty_weight = torch.where(
                    theta_conf < 0.1, torch.ones_like(theta_conf),
                    torch.where(theta_conf < 0.7, seg1,
                                torch.where(theta_conf < 0.9, seg2, seg3))
                ).unsqueeze(-1)  # [B,11,1]

                penalty_total = (penalty_weight * (dist ** 2)).mean()

                loss = j2d_loss.mean() + hand_j2d_loss.mean() + 1.0 * rot_reg_loss + 1000.0 * penalty_total + 1000.0 * delta_penalty

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([global_rot, selected_joint_rot, cam_full_raw], max_norm=1.0)
                optimizer.step()

                pbar.set_description(f"Optimization Iter {i}")
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        theta[:, theta_index, :] = selected_joint_rot
        optimized_theta = torch.cat((global_rot[:, None, :], theta[:, 1:, :]), dim=1)
        optimized_cam = torch.sigmoid(cam_full_raw) * (upper - lower) + lower

        return optimized_theta.detach(), beta_copy, optimized_cam.detach()

    def get_model(self, model_path, smpl_mean_path):
        """
        Load the model's weights.
        """
        predictor = ErgoNetPredictor(smpl_mean_path).to(self.device)
        predictor.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device), strict=True)
        return predictor

    def process_input(self,images,img_names,camera_intrinsics,poi=None,poi_type='bbox'):
        """
        Process the input data.
        """
        norm_img_final = []
        center_final = []
        scale_final = []
        crop_ul_final = []
        crop_br_final = []
        bboxes_final = []
        body_keypoints2d_final = []
        hand_keypoints2d_final = []
        image_shape_final = []
        camera_intrinsics_tensor_final = []
        data_id = []
        os.chdir(self.root_folder_path + '/checkpoints')
        person_detector = RFDETRBase(pretrain_weights="rf-detr-base.pth")
        person_detector.optimize_for_inference()
        os.chdir(self.root_folder_path)
        for i in range(len(images)):
            img = images[i]

            detection_results = person_detector.predict(img, threshold=0.7)
            raw_bboxes = np.concatenate(
                [detection_results.xyxy[detection_results.class_id == 1],
                 detection_results.confidence[detection_results.class_id == 1][:, None]],
                axis=1)
            if len(raw_bboxes) == 0:
                print("no person is being detected in", img_names[i])
                continue

            if poi != None:
                best_idx, best_score = None, -1
                for j in range(len(raw_bboxes)):
                    raw_bbox = raw_bboxes[j]
                    if poi_type == 'bbox':
                        x1, y1, x2, y2 = max(raw_bbox[0], poi[i][0]), max(raw_bbox[1], poi[i][1]), min(
                            raw_bbox[2], poi[i][2]), min(raw_bbox[3], poi[i][3])
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        area1 = (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1])
                        area2 = (poi[i][2] - poi[i][0]) * (poi[i][3] - poi[i][1])
                        score = inter / (area1 + area2 - inter + 1e-6)
                    elif poi_type == 'aoi':  # Intersection area
                        x1, y1, x2, y2 = max(raw_bbox[0], poi[i][0]), max(raw_bbox[1], poi[i][1]), min(
                            raw_bbox[2], poi[i][2]), min(raw_bbox[3], poi[i][3])
                        score = max(0, x2 - x1) * max(0, y2 - y1)
                    else:
                        raise ValueError(f"Unknown reference_type: {poi}")
                    if score > best_score:
                        best_score, best_idx = score, j
                if best_score == 0:
                    continue
                raw_bboxes = raw_bboxes[best_idx,:]
                raw_bboxes = np.expand_dims(raw_bboxes, axis=0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = img.shape

            keypoints = self.PoseEstimator2D.inference(img, raw_bboxes)
            enlarged_bboxes = self.PoseEstimator2D._tracker_res

            keypoints = np.asarray(list(keypoints.values()))
            hand_keypoints = keypoints[:,91:,[1,0,2]]
            body_keypoints = keypoints[:,:23,[1,0,2]]
            hand_keypoints = np.delete(hand_keypoints, [0, 21], axis=1)

            bboxes = enlarged_bboxes[0]
            bboxes_xywh = []
            for bbox in bboxes:
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                bboxes_xywh.append(bbox_xywh)
            bboxes_xywh = np.asarray(bboxes_xywh)
            all_norm_img = []
            all_center = []
            all_scale = []
            all_crop_ul = []
            all_crop_br = []

            for bbox in bboxes_xywh:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                norm_img, center, scale, crop_ul, crop_br, crop_img = process_image(img, [x, y, x + w, y + h])
                all_norm_img.append(norm_img)
                all_center.append(center.numpy())
                all_scale.append(scale)
                all_crop_ul.append(crop_ul)
                all_crop_br.append(crop_br)
                data_id.append(i)

            all_norm_img = np.asarray(all_norm_img)
            all_center = np.asarray(all_center)
            all_scale = np.asarray(all_scale)
            all_crop_ul = np.asarray(all_crop_ul)
            all_crop_br = np.asarray(all_crop_br)

            norm_img_single = torch.from_numpy(all_norm_img).float()
            center_single = torch.from_numpy(all_center).float()
            scale_single = torch.from_numpy(all_scale).float()
            crop_ul_single = torch.from_numpy(all_crop_ul).float()
            crop_br_single = torch.from_numpy(all_crop_br).float()
            bboxes_single = torch.from_numpy(bboxes_xywh).float()
            body_keypoints2d_single = torch.from_numpy(body_keypoints).float()
            hand_keypoints2d_single = torch.from_numpy(hand_keypoints).float()
            image_shape_single = torch.from_numpy(np.asarray([img_h,img_w])).repeat(bboxes_single.size()[0], 1).float()
            camera_intrinsics_tensor_single = torch.from_numpy(np.asarray(camera_intrinsics[i])).repeat(bboxes_single.size()[0], 1).float()

            norm_img_final.append(norm_img_single)
            center_final.append(center_single)
            scale_final.append(scale_single)
            crop_ul_final.append(crop_ul_single)
            crop_br_final.append(crop_br_single)
            bboxes_final.append(bboxes_single)
            body_keypoints2d_final.append(body_keypoints2d_single)
            hand_keypoints2d_final.append(hand_keypoints2d_single)
            image_shape_final.append(image_shape_single)
            camera_intrinsics_tensor_final.append(camera_intrinsics_tensor_single)

        if len(norm_img_final) == 0:
            return None, None, None,None,None,None,None,None, None,None,None

        norm_img_final = torch.cat(norm_img_final, dim=0).to(self.device)
        center_final = torch.cat(center_final, dim=0).to(self.device)
        scale_final = torch.cat(scale_final, dim=0).to(self.device)
        crop_ul_final = torch.cat(crop_ul_final, dim=0).to(self.device)
        crop_br_final = torch.cat(crop_br_final, dim=0).to(self.device)
        bboxes_final = torch.cat(bboxes_final, dim=0).to(self.device)
        body_keypoints2d_final = torch.cat(body_keypoints2d_final, dim=0).to(self.device)
        hand_keypoints2d_final = torch.cat(hand_keypoints2d_final, dim=0).to(self.device)
        image_shape_final = torch.cat(image_shape_final, dim=0).to(self.device)
        camera_intrinsics_tensor_final = torch.cat(camera_intrinsics_tensor_final, dim=0).to(self.device)

        return (norm_img_final, center_final, scale_final,crop_ul_final,crop_br_final,bboxes_final,body_keypoints2d_final,hand_keypoints2d_final,
                image_shape_final,camera_intrinsics_tensor_final,data_id)

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

    def check_input(self,images, img_names, camera_intrinsics, camera_distortion, gender, poi, poi_type):
        """
        Check if the inputs are in the required format.
        """
        images_check =  isinstance(images, list) and all(
            isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
            for img in images
        )
        if images_check is False:
            return False

        num_images = len(images)
        if poi == None:
            input_dims = [len(images), len(img_names), len(camera_intrinsics), len(camera_distortion)]
        else:
            input_dims = [len(images), len(img_names), len(camera_intrinsics), len(camera_distortion), len(poi)]
        if any(x != num_images for x in input_dims):
            return False

        if poi != None:
            if not (isinstance(poi, list) and all(isinstance(sub, list) and len(sub) == 4 for sub in poi)):
                return False

        if poi_type not in ['bbox','aoi']:
            return False

        if gender not in ['neutral','male','female']:
            return False

        return True

    def run(self,images, img_names, camera_intrinsics, camera_distortion, gender='neutral',
            poi=None, poi_type='bbox', joint_optimize=True):
        """
        Run the ErgoNet pipeline on the input images and return the predictions.

        args:
            images (list): raw, unprocessed input images.
            img_names (list): image names corresponding to the provided images.
            camera_intrinsics (tuple or None): camera intrinsic parameters (fx, fy, cx, cy).
                - fx: focal length along the x-axis (in pixels)
                - fy: focal length along the y-axis (in pixels)
                - cx: principal point along the x-axis (in pixels)
                - cy: principal point along the y-axis (in pixels)
                pass None if intrinsics are unknown.
            camera_distortion (list or None): camera distortion coefficients [k1, k2, p1, p2, k3].
                - k1, k2, k3: radial distortion coefficients
                - p1, p2: tangential distortion coefficients
                pass None if distortion parameters are unknown.
            poi: data used to identify the person of interest.
            poi_type (str): method for identifying the person of interest:
                - 'bbox': [x1, y1, x2, y2] — top-left and bottom-right coordinates of the bounding box.
                - 'aoi': [x1, y1, x2, y2] — top-left and bottom-right coordinates of the area of interest.
            gender (str): gender setting for SMPL-X model. one of 'neutral', 'male', or 'female'.
            joint_optimize (bool): if true, refine ErgoNet predictions using ViTPose joint predictions.

        returns:
            results: a dictionary containing ErgoNet's predicted results.
        """
        if self.check_input(images, img_names, camera_intrinsics, camera_distortion, gender, poi, poi_type) is False:
            raise ValueError(f"Inputs are not in the required format.")

        num_images = len(images)

        for i in range(len(images)):
            img = images[i]
            raw_img_h, raw_img_w, _ = img.shape
            if camera_intrinsics[i] is None:
                focal_length = (raw_img_w * raw_img_w + raw_img_h * raw_img_h) ** 0.5
                camera_intrinsics[i] = [focal_length, focal_length, raw_img_w / 2, raw_img_h / 2]
            if camera_distortion[i] is not None:
                camera_matrix = np.asarray([[camera_intrinsics[i][0], 0, camera_intrinsics[i][2]],
                                             [0, camera_intrinsics[i][1], camera_intrinsics[i][3]],
                                             [0, 0, 1]])
                img = cv2.undistort(img,camera_matrix,camera_distortion[i])
                images[i] = img

        with torch.no_grad():
            norm_img, center, scale,crop_ul,crop_br,bboxes,body_keypoints2d,hand_keypoints2d,image_shape,camera_intrinsics_tensor,data_id = self.process_input(images,img_names,camera_intrinsics,poi,poi_type)
            if norm_img is None:
                return None

            batch_size = norm_img.size(0)
            smplx_model = smplx.create(self.smpl_path, model_type='smplx',
                                      gender=gender, num_betas=10,
                                      ext='pkl',
                                      batch_size=batch_size).to(self.device)

            img_w = image_shape[:,1]
            img_h = image_shape[:,0]
            focal_length = camera_intrinsics_tensor[:,0:2]
            focal_length = focal_length.mean(dim=1)

            cx, cy, b = center[:, 0], center[:, 1], torch.tensor(scale * 200)
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

            pred_rotmat, pred_shape, pred_cam_crop, pred_pose_6d = self.predictor(norm_img, bbox_info)
            pred_cam_full = cam_crop2full_raw(pred_cam_crop, center, scale, image_shape, focal_length)
            pred_axis_angle = matrix_to_axis_angle(pred_rotmat)

            intrinsics_matrix = self.k_from_params(camera_intrinsics_tensor)
            body_kp2d = body_keypoints2d[:,:,0:2]
            body_kp2d_conf = body_keypoints2d[:,:,-1]
            hand_kp2d = hand_keypoints2d[:,:,0:2]
            hand_kp2d_conf = hand_keypoints2d[:,:,-1]

        if joint_optimize == True:
            theta, beta, cam_transl = self.optimize_joint(pred_axis_angle[:,0:22,:], pred_shape, pred_cam_full, intrinsics_matrix,
                           smplx_model, body_kp2d, body_kp2d_conf, hand_kp2d, hand_kp2d_conf,num_iter=1000)
        else:
            theta = pred_axis_angle[:,0:22,:]
            beta = pred_shape
            cam_transl = pred_cam_full

        with torch.no_grad():
            output = smplx_model(
                betas=beta,
                return_verts=True,
                body_pose=theta[:, 1:22, :],
                global_orient=theta[:,0,:],
                transl=cam_transl
            )
            j3d = output.joints
            j2d = projection(j3d,intrinsics_matrix)
            vertices = output.vertices.cpu().numpy().squeeze()

            smplx2openpose_joints = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                                 65]

            j3d = j3d[:,smplx2openpose_joints,:]
            j2d = j2d[:,smplx2openpose_joints,:]

            cam_transl = cam_transl.cpu().numpy().squeeze()
            j2d = j2d.cpu().numpy().squeeze()
            j3d = j3d.cpu().numpy().squeeze()
            theta = theta.cpu().numpy().squeeze()
            beta = beta.cpu().numpy().squeeze()
            bboxes = bboxes.cpu().numpy().squeeze()
            vitpose_j2d = body_keypoints2d.cpu().numpy().squeeze()

        if batch_size == 1:
            cam_transl = np.expand_dims(cam_transl, axis=0)
            j3d = np.expand_dims(j3d, axis=0)
            j2d = np.expand_dims(j2d, axis=0)
            theta = np.expand_dims(theta, axis=0)
            beta = np.expand_dims(beta, axis=0)
            vertices = np.expand_dims(vertices, axis=0)
            bboxes = np.expand_dims(bboxes, axis=0)
            vitpose_j2d = np.expand_dims(vitpose_j2d, axis=0)

        if poi != None:
            camera_intrinsics = [camera_intrinsics[i] for i in data_id]
            camera_distortion = [camera_distortion[i] for i in data_id]

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
            'num_images': num_images,
            'gender': gender,
            'data_id': data_id
        }
        return results