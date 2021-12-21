MODEL='ours'
RESULTS_DIR='/vulcan/scratch/tjoll/pc/rc3/results/point_cloud/'
FILE='pix3d_seen_depth'
python3 -W ignore fscore.py --model_name $MODEL  --pcd_dir "${RESULTS_DIR}/${FILE}/5_icp" --test_kind "${FILE}"
