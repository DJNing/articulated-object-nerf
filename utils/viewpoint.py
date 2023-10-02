import numpy as np
import math


conversion_matrix = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])

conversion_matrix_4x4 = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

def convert_ori_dir(origin, direction):
    dirs, ori = np.dot(conversion_matrix, direction), np.dot(conversion_matrix, origin)
    return ori, dirs

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

def degrees_to_radians(degrees):
    radians = degrees * (math.pi / 180)
    return radians