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
conversion_matrix = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])
def normalize(v):
    return v / np.sqrt(np.sum(v**2))

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    k = normalize(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def change_apply_change_basis(A, T, P):
    """
    Perform change of basis, apply transformation, and change back to the original basis.

    Args:
    A (numpy.ndarray): The original matrix in the B1 basis.
    T (numpy.ndarray): The transformation matrix to be applied.
    P (numpy.ndarray): The change of basis matrix from B1 to B2.

    Returns:
    numpy.ndarray: The resulting matrix after the entire process, expressed in the B1 basis.
    """
    # Step 1: Change A to the B2 basis
    A_B2 = np.linalg.inv(P).dot(A).dot(P)
    
    # Step 2: Apply the transformation T in the B2 basis
    A_B2_transformed = T.dot(A_B2)
    
    # Step 3: Change A_B2_transformed back to the original B1 basis
    A_B1_transformed = P.dot(A_B2_transformed).dot(np.linalg.inv(P))
    
    return A_B1_transformed

def calculate_E2(E1, axis_position, axis_direction, angle_degrees):
    # Convert the angle from degrees to radians
    angle_radians = degrees_to_radians(angle_degrees)
    
    # Create a 3x3 rotation matrix around the axis
    R = get_rotation_axis_angle(axis_direction, angle_radians)
    # Create a 4x4 transformation matrix for the rotation
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R
    print(R)
    
    # Create a 4x4 transformation matrix for translation to the axis position
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = axis_position
    # Calculate E2 by combining translation and rotation with E1
    # E2 = np.dot(rotation_matrix, np.dot(rotation_matrix, E1))
    E2 = change_apply_change_basis(E1, rotation_matrix, translation_matrix)
    return E2


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
        # V2 = np.dot(np.dot(np.dot(T_joint, T_theta), np.linalg.inv(T_joint)), V1)
        V2_prime = np.dot(np.linalg.inv(T_joint), np.dot(T_theta, V1))
        V2 = np.dot(V2_prime, T_joint)

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
# Load JSON data
json_fname = './data_base/laptop/10211/mobility_v2.json'
meta = load_json_to_dict(json_fname)
for link in meta:
    if link['joint'] == 'hinge':
        print(link['id'])
        print(link['jointData'])
        origin =  link['jointData']['axis']['origin']
        direction =  link['jointData']['axis']['direction']
# Setup the scene
camera, asset, scene = scene_setup(urdf)
# %%
# point = random_point_in_sphere(5, theta_range=[-0.01 * math.pi, 0.01*math.pi], phi_range=[0.01*math.pi, 0.01*math.pi])

def point_in_sphere(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return x, y, z

point = point_in_sphere(5, 1.2*math.pi, degrees_to_radians(75))
mat44 = calculate_cam_ext(point)
print(point)
print(mat44)
# print(point)
# mat44 = np.eye(4)
# mat44[0, -1] = 4
# mat44[1, -1] = 2
# mat44[2, -1] = 1
camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
# Render image with default pose
render_image(camera, scene, './draft/test_view.png')
seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
seg_view = seg_labels[..., 1].astype(np.uint8)  # actor-level
view_0 = camera.get_model_matrix()
art_degree = 100

# Rotate the joint and render a new image
asset.set_qpos(degrees_to_radians(art_degree))
render_image(camera, scene, './draft/test_view_15.png')
seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
seg_view_15 = seg_labels[..., 1].astype(np.uint8)  # actor-level

# Calculate joint state and transformation
joint_state = asset.get_links()[-1].pose.to_transformation_matrix()
# v2 = transform_V1_to_V2(joint_state, mat44, degrees_to_radians(15))


dirs, ori = np.dot(conversion_matrix, direction), np.dot(conversion_matrix, origin)


v2 = calculate_E2(mat44, ori, dirs, art_degree)
# view_2 = calculate_E2(view_0, np.array(origin), np.array(direction), art_degree)
# Set the new joint configuration and render the image
asset.set_qpos(0)
camera.set_pose(sapien.Pose.from_transformation_matrix(v2))
render_image(camera, scene, './draft/test_view_15_v2.png')
view_1 = camera.get_model_matrix()
# Overlay images
rgba_15_pil = Image.open('./draft/test_view_15.png')
rgba_v2_pil = Image.open('./draft/test_view_15_v2.png')
overlayed = overlay_images(rgba_15_pil, rgba_v2_pil, 0.5)
overlayed.save('./draft/overlayed_view.png')


rgba_pil = Image.open('./draft/test_view.png')
overlayed = overlay_images(rgba_pil, rgba_15_pil, 0.5)
overlayed.save('./draft/overlayed_art.png')

overlayed = overlay_images(rgba_pil, rgba_v2_pil, 0.5)
overlayed.save('./draft/overlayed_base.png')
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
def rotate_camera_extrinsic(e1, axis_direction, axis_origin, theta):
    try:
        # Create a 3x3 rotation matrix R based on the axis and angle
        u, v, w = axis_direction
        x, y, z = axis_origin
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        one_minus_cos = 1 - cos_theta

        R = np.array([
            [cos_theta + u**2 * one_minus_cos, u * v * one_minus_cos - w * sin_theta, u * w * one_minus_cos + v * sin_theta],
            [v * u * one_minus_cos + w * sin_theta, cos_theta + v**2 * one_minus_cos, v * w * one_minus_cos - u * sin_theta],
            [w * u * one_minus_cos - v * sin_theta, w * v * one_minus_cos + u * sin_theta, cos_theta + w**2 * one_minus_cos]
        ])

        # Create a 4x4 transformation matrix T to apply the rotation at the specified origin
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        # Calculate e2 by multiplying T with e1
        e2 = np.dot(T, e1)

        return e2

    except Exception as e:
        print(f"Error calculating e2: {str(e)}")
        return None
# %%
# point = point_in_sphere(5, math.pi, degrees_to_radians(90+15))
# mat44 = calculate_cam_ext(point)
# print(point)
# print(mat44)
# mat44[2, -1] += 2
# print(point)
# mat44 = np.eye(4)
# mat44[0, -1] = 4
# mat44[1, -1] = 2
# mat44[2, -1] = 1
conversion_matrix = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])
mat_new = rotate_camera_extrinsic(mat44, np.dot(conversion_matrix, direction), np.dot(conversion_matrix, origin), degrees_to_radians(-15))
camera.set_pose(sapien.Pose.from_transformation_matrix(mat_new))
# Render image with default pose
render_image(camera, scene, './draft/test_view_new.png')
rgba_new_pil = Image.open('./draft/test_view_new.png')
# %%
result = overlay_images(rgba_15_pil, rgba_new_pil, 0.5)
result.save('./draft/overlayed_new.png')
# %%
def calculate_E2(E1, axis_position, axis_direction, angle_degrees):
    # Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Create a 3x3 rotation matrix around the axis
    R = np.eye(3) + np.sin(angle_radians) * np.cross(np.eye(3), axis_direction) + \
        (1 - np.cos(angle_radians)) * np.outer(axis_direction, axis_direction)
    
    # Create a 4x4 transformation matrix for the rotation
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R
    
    # Create a 4x4 transformation matrix for translation to the axis position
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = axis_position
    
    # Calculate E2 by combining translation and rotation with E1
    E2 = np.dot(rotation_matrix, np.dot(translation_matrix, E1))
    
    return E2

# %%
links[-1].__dir__()
# %%

camera.set_pose(sapien.Pose.from_transformation_matrix(v2))
conversion_matrix_4x4 = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

ext_diff = camera.get_extrinsic_matrix() - np.dot(conversion_matrix_4x4, np.linalg.inv(v2)).astype(np.float32)
ext_diff[abs(ext_diff) < 1e-5] = 0

# %%
ext_diff
# %%
test_view = np.dot(conversion_matrix_4x4, np.linalg.inv(camera.get_extrinsic_matrix())).astype(np.float32)
# %%
test_view
# %%
viewpoint = camera.get_model_matrix()
# %%
viewpoint
# %%
ext = camera.get_extrinsic_matrix()
inv_ext = np.linalg.inv(ext)
inv_ext[np.abs(inv_ext) < 1e-6] = 0
test = np.dot(conversion_matrix_4x4, np.linalg.inv(ext)).astype(np.float32)
test[abs(test)<1e-4] = 0
# %%
test

# %%
viewpoint[np.abs(viewpoint) < 1e-5] = 0
print(viewpoint)
# %%
inv_ext
# %%
def create_viewpoint_matrix(camera_pose):
    # Modify the camera pose matrix directly
    camera_pose[0, 1] *= -1  # Reverse sign of C01
    camera_pose[0, 2] *= -1  # Reverse sign of C02
    camera_pose[1, 0] *= -1  # Reverse sign of C10
    camera_pose[1, 2] *= -1  # Reverse sign of C12
    camera_pose[2, 0] *= -1  # Reverse sign of C20
    camera_pose[2, 1] *= -1  # Reverse sign of C21

    # Return the viewpoint matrix
    return camera_pose
# %%
create_viewpoint_matrix(v2)
# %%
viewpoint
# %%
np.dot(v2[:3, :3], conversion_matrix)
# %%
r1 = viewpoint[:3, :3]
# %%
r2 = v2[:3, :3]
# %%
np.dot(r1.T, r2)
# %%
r1
# %%
r2
# %%
column_swap = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])
sign_convertion = np.array([
    [1, -1, -1],
    [-1, -1, 1],
    [1, 1, 1]
])
r2_swap = np.dot(r2, column_swap)
print(r2_swap)
print(r1)
print(r2_swap * sign_convertion)
# %%
# Original matrix A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Define a permutation matrix for swapping columns 1 and 2
# In this case, we want to swap the first and second columns, so the permutation matrix is:
# P = [[0, 1, 0],
#      [1, 0, 0],
#      [0, 0, 1]]
P = np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]])

# Perform the column swap by multiplying A with the permutation matrix P
result = np.dot(A, P)

print(result)
# %%
np.dot(np.dot(sign_convertion, column_swap), r2)
# %%
r1
# %%
r2_swap
# %%
np.dot(sign_convertion, column_swap)
# %%
def pose2view(pose):
    '''
    pose: 4x4 matrix
    '''
    column_swap = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    sign_convertion = np.array([
        [1, -1, -1],
        [-1, -1, 1],
        [1, 1, 1]
    ])
    R = pose[:3, :3]
    view_R = np.dot(R, column_swap) * sign_convertion
    view = np.eye(4)
    view[:3, :3] = view_R
    view[:3, -1] = pose[:3, -1]
    return view

def view2pose(pose):
    '''
    pose: 4x4 matrix
    '''
    column_swap = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    sign_convertion = np.array([
        [1, -1, -1],
        [-1, -1, 1],
        [1, 1, 1]
    ])
    R = pose[:3, :3] * sign_convertion
    view_R = np.dot(R, column_swap) 
    view = np.eye(4)
    view[:3, :3] = view_R
    view[:3, -1] = pose[:3, -1]
    return view

# %%
pose2view(v2)
# %%
diff = v2 - view2pose(pose2view(v2))
# %%
diff[abs(diff) < 1e-5] = 0
# %%
diff
# %%

# %%
# learnable params: axis direction, axis position, quaternion
# E1 = view2pose(c2w) --> convert the input c2w to camera extrinsic
# learnable params: axis direction, axis position, quaternion
# E2 = calculate_art_ext(E1, axis direction, axis position, quaternion) --> implement calculate_E2 in pytorch
# c2w_art = pose2view(E2)
# %%
dirs
# %%
ori
# %%
