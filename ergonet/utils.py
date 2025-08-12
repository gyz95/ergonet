import torch
from collections import OrderedDict
from scipy.signal import butter, filtfilt
import numpy as np
import smplx

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def cam_crop2full_raw(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def cam_crop2full(crop_cam, center, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_w = full_img_shape[:, 0]
    img_h = full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], center[:, 2]
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def projection(points, intrinsicParam):
    # points: (batch, num_points, 3)
    # intrinsicParam: (batch, 3, 3)

    batch_size, num_points, _ = points.shape

    # Extract X, Y, Z
    X = points[..., 0]
    Y = points[..., 1]
    Z = points[..., 2].clamp(min=1e-6)  # avoid division by zero

    # Normalize image coordinates
    x = X / Z
    y = Y / Z

    # Get intrinsic parameters
    fx = intrinsicParam[:, 0, 0].unsqueeze(1)
    fy = intrinsicParam[:, 1, 1].unsqueeze(1)
    cx = intrinsicParam[:, 0, 2].unsqueeze(1)
    cy = intrinsicParam[:, 1, 2].unsqueeze(1)

    # Project to image plane
    u = fx * x + cx
    v = fy * y + cy

    pix_loc = torch.stack([u, v], dim=-1)  # (batch, num_points, 2)
    return pix_loc

def butter_low_pass_filter(data, cutoff_freq=0.3, sampling_freq=5, order=2):
    # Calculate the Nyquist frequency
    nyquist = 0.5 * sampling_freq

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each feature independently
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])

    return filtered_data


def smpl_joints_from_vertices(model_path, vertices, device, update_hips=True):
    """
    Compute OpenPose-style joints from SMPL-formatted vertices.
    """
    device = device
    smpl_model = smplx.SMPLLayer(model_path=model_path + '/smpl').to(device)
    J_regressor = smpl_model.J_regressor.to(device)  # (24, 6890)

    # Default SMPL to OpenPose joint mapping
    smpl_to_openpose = torch.tensor([
        24, 12, 17, 19, 21, 16, 18, 20,
        0, 2, 5, 8, 1, 4, 7,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34
    ], dtype=torch.long, device=device)

    # Compute SMPL joints from vertices
    joints = J_regressor @ vertices  # (B, 6890, 3) => (B, 24, 3)
    joints = smpl_model.vertex_joint_selector(vertices, joints)

    # Select mapped joints
    body_joints = joints[:, smpl_to_openpose]

    # Optional hip update
    if update_hips:
        body_joints[:, [9, 12]] = body_joints[:, [9, 12]] + \
                                  0.25 * (body_joints[:, [9, 12]] - body_joints[:, [12, 9]]) + \
                                  0.5 * (body_joints[:, [8]] - 0.5 * (
                body_joints[:, [9, 12]] + body_joints[:, [12, 9]]))

    return body_joints  # shape: (B, 25, 3)

class SMPLJointExtractor:
    def __init__(self, model_path, device):
        """
        Initialize the SMPL model and joint regressor once.
        """
        self.device = device
        self.smpl_model = smplx.SMPLLayer(model_path=model_path + '/smpl').to(device)
        self.J_regressor = self.smpl_model.J_regressor.to(device)

        # SMPL-to-OpenPose 25-joint mapping (including foot and hand keypoints)
        self.smpl_to_openpose = torch.tensor([
            24, 12, 17, 19, 21, 16, 18, 20,
            0, 2, 5, 8, 1, 4, 7,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34
        ], dtype=torch.long, device=device)

    def extract(self, vertices, update_hips=True):
        """
        Compute OpenPose-style joints from SMPL vertices.

        Args:
            vertices: Tensor of shape (B, 6890, 3)
            update_hips: Whether to apply pelvis-centered hip correction

        Returns:
            body_joints: Tensor of shape (B, 25, 3)
        """
        # Compute default SMPL joints
        joints = self.J_regressor @ vertices  # (B, 24, 3)
        joints = self.smpl_model.vertex_joint_selector(vertices, joints)

        # Select OpenPose-style joints
        body_joints = joints[:, self.smpl_to_openpose]

        if update_hips:
            # Optional hip refinement (same logic as your original)
            body_joints[:, [9, 12]] = body_joints[:, [9, 12]] + \
                0.25 * (body_joints[:, [9, 12]] - body_joints[:, [12, 9]]) + \
                0.5 * (body_joints[:, [8]] - 0.5 * (body_joints[:, [9, 12]] + body_joints[:, [12, 9]]))

        return body_joints