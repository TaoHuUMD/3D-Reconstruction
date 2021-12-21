import numpy as np
import math

USE_MITSUBA_RENDER_PARA = True

if USE_MITSUBA_RENDER_PARA:
    MAX_OBJ_SCALE = 2.0

#real depth to [0,0.96] -- pix*255
SCALED_REAL_DEPTH_MIN = 0.78
SCALED_REAL_DEPTH_MAX = 1.21
SCALED_REAL_SCALE_DEPTH = 0.41
SCALED_NOISE_SCALE  = SCALED_REAL_SCALE_DEPTH*0.03

#NO SCALE

REAL_DEPTH_MIN = MAX_OBJ_SCALE * SCALED_REAL_DEPTH_MIN
REAL_DEPTH_MAX = MAX_OBJ_SCALE * SCALED_REAL_DEPTH_MAX
REAL_SCALE_DEPTH = MAX_OBJ_SCALE * SCALED_REAL_SCALE_DEPTH

if USE_MITSUBA_RENDER_PARA:
    REAL_DEPTH_MIN = 2.0-0.58
    REAL_DEPTH_MAX = 2.0+0.58
    SCALED_REAL_SCALE_DEPTH = REAL_DEPTH_MAX - REAL_DEPTH_MIN
    REAL_SCALE_DEPTH = MAX_OBJ_SCALE * SCALED_REAL_SCALE_DEPTH
    SCALED_NOISE_SCALE  = SCALED_REAL_SCALE_DEPTH*0.03

    REFINED_WITH_BOX = True

NOISE_SCALE  = REAL_SCALE_DEPTH*0.03


TRAIN_DEPTH_MIN = 0.0
TRAIN_DEPTH_MAX = 0.99
TRAIN_SCALE_DEPTH = TRAIN_DEPTH_MAX - TRAIN_DEPTH_MIN


#projections
GRID_SIZE = int(5)

DEPTH_BOUND = 5.0

#background in training, -1*
EXR_COORD_BACKGROUND_VALUE = 0.8
EXR_COORD_BACKGROUND_SHIFT = 0.4

EXR_RGB_BACKGROUND_VALUE = 0.8
EXR_DEPTH_BACKGROUND_VALUE = 0.8

FIXED_VIEW_NUM = 8