#generate S_d
#single view reconstruction of seen and unseen categories on Pix3D dataset.

#models
PRETRAINED_NET_2D_3D='2D_3D_Obj8'
PRETRAINED_NET_COMPLETION='MDCN'

RESULTS_DIR='./results/point_cloud/'
GROUND_TRUTH_DIR='./datasets/pix3d/ground_truth/'

VOTE_NUM='5'

#seen category
SAVE_MODEL_NAME='pix3d_seen_depth'
NOC_DATA_PREFIX='pix3d_seen'

## 1. 2D to 3D
#This model takes input rgb images from datasets/pix3d/pix3d_seen/test and generates object coordinate images into ./results/pix3d_seen_coord/ directory and furtuer generate partial views (texture/depth pairs) into ./datasets/pix3d/"${NOC_DATA_PREFIX}_${PRETRAINED_NET_2D_3D}".
python3 test.py --dataroot  ./datasets/pix3d/pix3d_seen/ --name $PRETRAINED_NET_2D_3D --model Coord --dataset_mode reconstructeCoord  --norm batch --input_nc 3 --output_nc 3 --flag test --save_results 1 --make_coord_depth 1 --coord_data_prefix $NOC_DATA_PREFIX --gpu_ids 0 --gt_depth_dir ./datasets/pix3d/pix3d_seen/texture_depth --texture 1 --rgb_mask 1 --save_model_name pix3d_seen_coord --test_pix3d

## 2. multi-view completion.
#This model takes partial views and generates completed views into ./results/$SAVE_MODEL_NAME/.
python3 test.py --dataroot ./datasets/pix3d/"${NOC_DATA_PREFIX}_${PRETRAINED_NET_2D_3D}" --name $PRETRAINED_NET_COMPLETION --model RecShapeMemory  --dataset_mode TextureDepth --gpu_ids 0 --input_nc 1 --output_nc 1 --save_results 1 --norm batch --save_model_name $SAVE_MODEL_NAME --only_depth --test_pix3d

## 3. joint-fusion.
#Fuse multiple completed views from ./results/$SAVE_MODEL_NAME/. into point clouds stored in "${RESULTS_DIR}/${SAVE_MODEL_NAME}/${VOTE_NUM}".
python3 fusion.py --model_name $SAVE_MODEL_NAME  --vote_num $VOTE_NUM --no_texture

MODEL='ours'
#Calculate CD.
python3 -W ignore metrics.py --model_name $MODEL --pcd_dir "${RESULTS_DIR}/${SAVE_MODEL_NAME}/${VOTE_NUM}" --use_icp --ground_truth_dir $GROUND_TRUTH_DIR


#unseen category.
SAVE_MODEL_NAME='pix3d_unseen_depth'
NOC_DATA_PREFIX='pix3d_unseen'

## 1. 2D to 3D
python3 test.py --dataroot  ./datasets/pix3d/pix3d_seen/ --name $PRETRAINED_NET_2D_3D --model Coord --dataset_mode reconstructeCoord  --norm batch --input_nc 3 --output_nc 3 --flag test --save_results 1 --make_coord_depth 1 --coord_data_prefix $NOC_DATA_PREFIX --gpu_ids 0 --gt_depth_dir ./datasets/pix3d/pix3d_seen/texture_depth --texture 1 --rgb_mask 1 --save_model_name pix3d_seen_coord --test_pix3d

## 2. multi-view completion.
python3 test.py --dataroot ./datasets/pix3d/"${NOC_DATA_PREFIX}_${PRETRAINED_NET_2D_3D}" --name $PRETRAINED_NET_COMPLETION --model RecShapeMemory  --dataset_mode TextureDepth --gpu_ids 0 --input_nc 1 --output_nc 1 --save_results 1 --norm batch --save_model_name $SAVE_MODEL_NAME --only_depth --test_pix3d

## 3. joint-fusion.
python3 fusion.py --model_name $SAVE_MODEL_NAME  --vote_num $VOTE_NUM --no_texture

MODEL='ours'
python3 -W ignore metrics.py --model_name $MODEL --pcd_dir "${RESULTS_DIR}/${SAVE_MODEL_NAME}/${VOTE_NUM}" --use_icp --ground_truth_dir $GROUND_TRUTH_DIR



