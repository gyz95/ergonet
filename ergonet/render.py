import pyrender
import torch.nn.functional as F
import torch
import numpy as np
import trimesh
import PIL

def render(vertices, faces, raw_frame, Camera_Intrinic, image_shape):
    background_image = raw_frame
    vertices_to_render = vertices

    vertex_colors = np.ones([vertices_to_render.shape[0], 4]) * [0 / 255, 0 / 255, 255 / 255, 1]

    tri_mesh = trimesh.Trimesh(vertices_to_render, faces, vertex_colors=vertex_colors)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    tri_mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)

    camera = pyrender.IntrinsicsCamera(fx=Camera_Intrinic[0], fy=Camera_Intrinic[1],
                                       cx=Camera_Intrinic[2],
                                       cy=Camera_Intrinic[3])

    scene.add(camera, pose=camera_pose)

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)

    scene.add(light, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_shape[0],
        viewport_height=image_shape[1],
        point_size=1.0
    )
    rgb, rend_depth = renderer.render(scene)
    img = rgb.astype(np.float32) / 255.0

    blended_image = img[:, :, :3]
    blending_weight = 1.0
    if background_image is not None:
        background_image = torch.tensor(background_image, dtype=torch.float32) / 255.0

        # Rescale the rendering results to blend with the background image.
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format
        img_tensor = F.interpolate(img_tensor, size=(background_image.shape[0], background_image.shape[1]), mode='bilinear', align_corners=True)
        img_tensor = img_tensor.squeeze(0).permute(1, 2, 0)  # Convert back to HWC format

        # Blend the rendering result with the background image.
        rend_depth_tensor = torch.tensor(rend_depth).unsqueeze(0).unsqueeze(0)  # Convert to NCHW format
        foreground = (rend_depth_tensor > 0).float() * blending_weight
        foreground = F.interpolate(foreground, size=(background_image.shape[0], background_image.shape[1]), mode='bilinear', align_corners=True)
        foreground = foreground.squeeze(0).squeeze(0)  # Convert back to HW format

        foreground = foreground.unsqueeze(2)  # Convert to HWC format for broadcasting
        blended_image = (foreground * img_tensor[:, :, :3] + (1.0 - foreground) * background_image)

    return blended_image.numpy()

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    image = PIL.Image.fromarray(tensor)
    open_cv_image = np.asarray(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image