#generate S_{d+t} where d (depth) is from S_d and t (texture) is from S_{dt}.
# mix the depth of S_d and texture of S_{dt} together.
# the process is slow due to bfs.
## 3. joint-fusion.
DEPTH_MODEL_NAME='pix3d_seen_depth'
TEXTURE_MODEL_NAME_='pix3d_seen_texture'
python3 fusion.py --model_name $DEPTH_MODEL_NAME --texture_model_name $TEXTURE_MODEL_NAME_ --mix_texture_depth
exit
#unseen category
DEPTH_MODEL_NAME='pix3d_unseen_depth'
TEXTURE_MODEL_NAME='pix3d_unseen_texture'
python3 fusion.py --model_name $DEPTH_MODEL_NAME --texture_model_name $TEXTURE_MODEL_NAME --mix_texture_depth