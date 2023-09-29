# %%
import numpy as np
import sapien.core as sapien
from PIL import Image
import json
from datagen.data_utils import *
from pathlib import Path as P
from utils.visualization import overlay_images

# %%
# def setup_scene(urdf):
#     # Set up the SAPIEN scene and camera
#     camera, asset, scene = scene_setup(urdf_file=urdf)
#     return camera, asset, scene
def transform_V1_to_V2(joint_state, V1, delta_rotation):
    try:
        # Extract the rotation matrix R_joint from the joint state
        R_joint = joint_state[:3, :3]

        u = R_joint[0, 2]
        v = R_joint[1, 2]
        w = R_joint[2, 2]


        # Calculate the rotation matrix R_theta for the delta rotation along the joint axis
        cos_theta = np.cos(delta_rotation)
        sin_theta = np.sin(delta_rotation)
        R_theta = np.array([[cos_theta + u**2 * (1 - cos_theta), u * v * (1 - cos_theta) - w * sin_theta, u * w * (1 - cos_theta) + v * sin_theta],
                            [v * u * (1 - cos_theta) + w * sin_theta, cos_theta + v**2 * (1 - cos_theta), v * w * (1 - cos_theta) - u * sin_theta],
                            [w * u * (1 - cos_theta) - v * sin_theta, w * v * (1 - cos_theta) + u * sin_theta, cos_theta + w**2 * (1 - cos_theta)]])


        # Create 4x4 transformation matrices
        T_joint = np.eye(4)
        T_joint[:3, :3] = R_joint

        T_theta = np.eye(4)
        T_theta[:3, :3] = R_theta

        # Compute V2 by multiplying the transformation matrices
        V2 = np.dot(np.dot(np.dot(T_joint, T_theta), np.linalg.inv(T_joint)), V1)

        return V2

    except Exception as e:
        print(f"Error calculating V2: {str(e)}")
        return None
def degrees_to_radians(degrees):
    radians = degrees * (math.pi / 180)
    return radians
def render_image(camera, scene, save_path):
    # Render an image using the provided camera and save it to the specified path
    scene.step()
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_float_texture('Color')
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img, 'RGBA')
    rgba_pil.save(save_path)

def load_json_to_dict(fname):
    # Load JSON data from a file into a dictionary
    try:
        with open(fname, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON from {fname}: {str(e)}")
        return None
# %%
urdf = 'data_base/laptop/10211/mobility.urdf'

# Setup the scene
camera, asset, scene = scene_setup(urdf)
# %%
point = random_point_in_sphere(5, theta_range=[1 * math.pi, 1.5*math.pi], phi_range=[0.4*math.pi, 0.5*math.pi])
mat44 = calculate_cam_ext(point)
camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
# Render image with default pose
render_image(camera, scene, './draft/test_view.png')

# Rotate the joint and render a new image
asset.set_qpos(degrees_to_radians(15))
render_image(camera, scene, './draft/test_view_15.png')

# Calculate joint state and transformation
joint_state = asset.get_links()[-1].pose.to_transformation_matrix()
v2 = transform_V1_to_V2(joint_state, mat44, degrees_to_radians(15))

# Set the new joint configuration and render the image
asset.set_qpos(0)
camera.set_pose(sapien.Pose.from_transformation_matrix(v2))
render_image(camera, scene, './draft/test_view_15_v2.png')

# Overlay images
rgba_15_pil = Image.open('./draft/test_view_15.png')
rgba_v2_pil = Image.open('./draft/test_view_15_v2.png')
overlayed = overlay_images(rgba_15_pil, rgba_v2_pil, 0.5)
overlayed.save('./draft/overlayed_view.png')


rgba_pil = Image.open('./draft/test_view.png')
overlayed = overlay_images(rgba_pil, rgba_15_pil, 0.5)
overlayed.save('./draft/overlayed_art.png')
# %%
# Load JSON data
json_fname = './data_base/laptop/10211/mobility_v2.json'
meta = load_json_to_dict(json_fname)
for link in meta:
    if link['joint'] == 'hinge':
        print(link['id'])
        print(link['jointData'])
        origin =  link['jointData']['axis']['origin']
        direction =  link['jointData']['axis']['direction']


# %%
links = asset.get_links()
for link in links:
    print(link.id)
# %%
links[-1].__dir__()
joints = asset.get_joints()
# %%
joints[-1].__dir__()
# %%
joints[-1].get_limits()/math.pi * 180
# %%
def transform_V1_to_V2_new(axis_direction, axis_origin, V1, delta_rotation):
    try:
        # Calculate the rotation matrix R_theta for the delta rotation along the joint axis
        cos_theta = np.cos(delta_rotation)
        sin_theta = np.sin(delta_rotation)

        # Extract the components of the axis direction
        u, v, w = axis_direction

        # Calculate the rotation matrix R_theta for the delta rotation along the joint axis
        R_theta = np.array([[cos_theta + u**2 * (1 - cos_theta), u * v * (1 - cos_theta) - w * sin_theta, u * w * (1 - cos_theta) + v * sin_theta],
                            [v * u * (1 - cos_theta) + w * sin_theta, cos_theta + v**2 * (1 - cos_theta), v * w * (1 - cos_theta) - u * sin_theta],
                            [w * u * (1 - cos_theta) - v * sin_theta, w * v * (1 - cos_theta) + u * sin_theta, cos_theta + w**2 * (1 - cos_theta)]])

        # Create a 4x4 transformation matrix T_joint representing the joint transformation
        T_joint = np.eye(4)
        T_joint[:3, :3] = R_theta

        # Set the translation part of T_joint based on the axis origin
        T_joint[:3, 3] = axis_origin

        # Compute V2 by multiplying the transformation matrices
        V2 = np.dot(T_joint, V1)

        return V2

    except Exception as e:
        print(f"Error calculating V2: {str(e)}")
        return None
# %%
v2 = transform_V1_to_V2_new(direction, origin, mat44, degrees_to_radians(-15))

# Set the new joint configuration and render the image
asset.set_qpos(0)
camera.set_pose(sapien.Pose.from_transformation_matrix(v2))
render_image(camera, scene, './draft/test_view_15_v2.png')
rgba_v2_pil = Image.open('./draft/test_view_15_v2.png')
overlayed = overlay_images(rgba_15_pil, rgba_v2_pil, 0.5)
overlayed.save('./draft/overlayed_view.png')
# %%
