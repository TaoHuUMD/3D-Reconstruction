import numpy as np
import torch
from . import global_variables

def get_fixed_8_views():
    if global_variables.USE_MITSUBA_RENDER_PARA:
        return get_fixed_8_views_mitsuba()
    
def lookAt(p):
    """ Compute LookAt matrix
        with lookat = [0; 0; 0]
             up     = [0; 1; 0]
    """
    t = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    d = (t-p) / np.linalg.norm(p)
    left = np.cross(up, d)
    left = left / np.linalg.norm(left)
    newUp = np.cross(d, left)
    
    result = np.zeros([4, 4], dtype=np.float32)
    result[0:3,0] = left
    result[0:3,1] = newUp
    result[0:3,2] = d
    result[0:3,3] = p
    result[3,3] = 1
    return result


def single_view_random_pose():

    SCALE=global_variables.MAX_OBJ_SCALE

    num = 1

    angles=[]

    for i in range(num):
        angles.append([np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi])
        
    all_poses = []

    translate = []

    for i in range(num):
        angle_x = angles[i][0]
        angle_y = angles[i][1]
        angle_z = angles[i][2]
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        # Set camera pointing to the origin and 1 unit away from the origin
    
        #t = np.array([[positions[i][0]],[positions[i][1]],[positions[i][2]]])
        t = np.copy(np.expand_dims(R[:, 2], 1))
        
        t=t*2.7

        translate.append(t[0][0]*t[0][0]+ t[1][0]*t[1][0] + t[2][0]*t[2][0])

        all_poses.append(np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0))
        #print (all_poses[-1])
    return all_poses[0]


def get_fixed_intrinsics():
    if global_variables.USE_MITSUBA_RENDER_PARA:
        return get_rgb_intrinsics()
    return np.array([[640, 0, 128], [0, 640, 128], [0, 0, 1]])

def get_rgb_intrinsics():
    return np.array([[480, 0, 128], [0, 480, 128], [0, 0, 1]])

def get_fixed_8_views_mitsuba():

    all_poses = []

    camera_loc = np.array([    
    [1.15470054,  -1.15470054, -1.15470054        ],
    [1.15470054,  -1.15470054,  1.15470054        ],
    [1.15470054,  1.15470054,   1.15470054        ],
    [1.15470054,  1.15470054, -1.15470054        ],
    [-1.15470054,  1.15470054, -1.15470054       ],
    [-1.15470054,  1.15470054,  1.15470054       ],
    [-1.15470054, -1.15470054,  1.15470054       ],
    [-1.15470054, -1.15470054, -1.15470054       ]
    ]
    )

    for i in range(camera_loc.shape[0]):
        all_poses.append(lookAt(camera_loc[i]))

    return all_poses

    all_poses.append([[0.7071067690849304, 0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[-0.0000000000000000, 0.8164966106414795, 0.5773502588272095, -1.1547005176544189],[0.7071067690849304, -0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

    all_poses.append([[-0.7071067690849304, 0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.8164966106414795, 0.5773502588272095, -1.1547005176544189],[0.7071067690849304, 0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    all_poses.append([[-0.7071067690849304, -0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.8164966106414795, -0.5773502588272095, 1.1547005176544189],[0.7071067690849304, -0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    all_poses.append([[0.7071067690849304, -0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[-0.0000000000000000, 0.8164966106414795, -0.5773502588272095, 1.1547005176544189],[0.7071067690849304, 0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    all_poses.append([[0.7071067690849304, 0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.8164966106414795, -0.5773502588272095, 1.1547005176544189],[-0.7071067690849304, 0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    all_poses.append([[-0.7071067690849304, 0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.8164966106414795, -0.5773502588272095, 1.1547005176544189],[-0.7071067690849304, -0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

    all_poses.append([[-0.7071067690849304, -0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.8164966106414795, 0.5773502588272095, -1.1547005176544189],[-0.7071067690849304, 0.4082483053207397, -0.5773502588272095, 1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    
    all_poses.append([[0.7071067690849304, -0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.8164966106414795, 0.5773502588272095, -1.1547005176544189],[-0.7071067690849304, -0.4082483053207397, 0.5773502588272095, -1.1547005176544189],[0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

    return all_poses


def normalize_depth_(img_gpu):
    
    img = img_gpu.cpu().numpy()
    h,w =img.shape

    max_depth = -100
    min_depth = 100

    for i in range(h):
        for j in range(w):
            if img[i][j] > 10:
                continue
            if img[i][j] > max_depth:
                max_depth = img[i][j]
            if img[i][j] < min_depth:
                min_depth = img[i][j]

    for i in range(h):
        for j in range(w):
            if img[i][j] > 10:
                img[i][j] = 1
            else:
                img[i][j] = (img[i][j] - min_depth)*global_data_attributes.TRAIN_SCALE_DEPTH/(max_depth-min_depth) + global_data_attributes.TRAIN_DEPTH_MIN

    return (img)#.type(torch.uint8)


def normalize_depth_min_max(img_gpu, min_v, max_v, DEPTH_BOUND=10.0):
    
    img = img_gpu.cpu().numpy()
    h,w =img.shape

    for i in range(h):
        for j in range(w):
            if img[i][j] > DEPTH_BOUND:
                img[i][j] = 1
            else:
                img[i][j] = (img[i][j] - min_v)*global_data_attributes.TRAIN_SCALE_DEPTH/(max_v-min_v) + global_data_attributes.TRAIN_DEPTH_MIN

    return img 

 
def normalize_pc_to_pix(points):
    
    min_v = torch.min(points)
    max_v = torch.max(points)

    pix_pc = (points-min_v)*255/(max_v-min_v) + 0
    return pix_pc.type(torch.uint8)