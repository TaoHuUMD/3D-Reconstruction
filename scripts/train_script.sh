
#training on multiple categories.

PRETRAINED_NET_2D_3D_GOOD='2D_3D_Obj8'
PRETRAINED_NET_2D_3D_BAD='2D_3D_Obj1'
PRETRAINED_NET_MDCN='MDCN'
PRETRAINED_NET_MTDCN='MTDCN'


#2D-3D transformation. For each object, one view is selected in training.
python3 train.py --dataroot ./datasets/all/coord/1/ --weight_l_loss 1 --name $PRETRAINED_NET_2D_3D_BAD --model Coord --lr 0.0009 --batch_size 64  --dataset_mode reconstructeCoord --direction AtoB --save_epoch_freq 1 --gpu_ids 0,1,2,3 --input_nc 3 --output_nc 3 --display_id 0 --num_threads 10

#Test all the 8 random views of each object to generate training dataset for the view completion stage.
python3 test.py --dataroot  ./datasets/all/coord/8/ --name $PRETRAINED_NET_2D_3D_BAD --model Coord --dataset_mode reconstructeCoord  --norm batch --input_nc 3 --output_nc 3 --flag train --save_results 0 --make_coord_depth 1 --coord_data_prefix training_data --gpu_ids 0 --gt_depth_dir ./datasets/all/texture_depth --texture 1

#MDCN
python3 train.py --dataroot ./datasets/all/coord/training_data_2D_3D_Obj1/ --weight_l_loss 1 --name $PRETRAINED_NET_MDCN  --model RecShapeMemory --lr 0.0012 --batch_size 128  --dataset_mode TextureDepth --direction AtoB --save_epoch_freq 1 --gpu_ids 0,1,2,3 --input_nc 1 --output_nc 1 --display_id 0 --num_threads 12 --only_depth

#MTDCN
python3 -W ignore train.py --dataroot ./datasets/all/coord/training_data_2D_3D_Obj1/ --weight_l_loss 1 --name $PRETRAINED_NET_MTDCN  --model RecTextureDepth --lr 0.0008 --batch_size 48 --dataset_mode TextureDepth --direction AtoB --save_epoch_freq 1 --gpu_ids 0,1,2,3 --input_nc 4 --output_nc 4 --display_id 0 --num_threads 12