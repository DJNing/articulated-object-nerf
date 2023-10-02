import numpy as np
import math
import torch

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

def change_apply_change_basis_torch(A, T, P):
    """
    Perform change of basis, apply transformation, and change back to the original basis.

    Args:
    A (torch.Tensor): The original matrix in the B1 basis.
    T (torch.Tensor): The transformation matrix to be applied.
    P (torch.Tensor): The change of basis matrix from B1 to B2.

    Returns:
    torch.Tensor: The resulting matrix after the entire process, expressed in the B1 basis.
    """
    # Step 1: Change A to the B2 basis
    A_B2 = torch.matmul(torch.matmul(torch.inverse(P), A), P)
    # Step 2: Apply the transformation T in the B2 basis
    A_B2_transformed = torch.matmul(T, A_B2)
    
    # Step 3: Change A_B2_transformed back to the original B1 basis
    A_B1_transformed = torch.matmul(torch.matmul(P, A_B2_transformed), torch.inverse(P))
    
    return A_B1_transformed

def calculate_E2(E1, axis_position, axis_direction, angle_degrees):
    '''
    calculate the extrinsic params for camera
    '''
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

def view2pose(view):
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
    R = view[:3, :3] * sign_convertion
    pose_R = np.dot(R, column_swap) 
    pose = np.eye(4)
    pose[:3, :3] = pose_R
    pose[:3, -1] = view[:3, -1]
    return view

def pose2view_torch(pose):
    """
    Convert a 4x4 pose matrix to a 4x4 view matrix with differentiable operations.

    Args:
    pose (torch.Tensor): 4x4 pose matrix.

    Returns:
    torch.Tensor: 4x4 view matrix.
    """
    column_swap = torch.tensor([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float32).to(pose)
    
    sign_conversion = torch.tensor([
        [1, -1, -1],
        [-1, -1, 1],
        [1, 1, 1]
    ], dtype=torch.float32).to(pose)

    R = pose[:3, :3]
    view_R = torch.matmul(R, column_swap) * sign_conversion
    view = torch.eye(4, dtype=torch.float32).to(pose)
    view[:3, :3] = view_R
    view[:3, -1] = pose[:3, -1]

    return view

def view2pose_torch(view):
    '''
    view: 4x4 matrix (torch.Tensor)
    '''
    column_swap = torch.Tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]).to(view)
    sign_conversion = torch.Tensor([
        [1, -1, -1],
        [-1, -1, 1],
        [1, 1, 1]
    ]).to(view)
    
    # Extract the rotation matrix R from the view matrix and apply sign conversion
    R = view[:3, :3] * sign_conversion
    
    # Compute the pose rotation matrix by applying column swapping
    pose_R = torch.matmul(R, column_swap)
    
    # Initialize the pose matrix as an identity matrix
    pose = torch.eye(4).to(view)
    
    # Set the pose rotation matrix and translation vector in the pose matrix
    pose[:3, :3] = pose_R
    pose[:3, -1] = view[:3, -1]
    
    return pose

