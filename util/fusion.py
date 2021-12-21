import argparse, sys, os
import open3d
import projections.operations as operations
import config

if __name__ == '__main__':

    print('fusion')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, required=True)
    parser.add_argument('--vote_num', type = int, default=5)
    parser.add_argument('--base_path', type = str, default = config.BASE_PATH)
    parser.add_argument('--no_texture', action='store_true')
    parser.add_argument('--mix_texture_depth', action='store_true')
    #parser.add_argument('--mix_texture_depth', action='store_true')
    parser.add_argument('--texture_model_name', type = str)
    args = parser.parse_args()

    print(args.mix_texture_depth,'cmd')

    fus = operations.Depth_Fusion_PostProcessing(gpu_id=0, reso=256, view_num=8, vote_number=args.vote_num, base_path=args.base_path)
    
    if args.no_texture:
        #S_d
        fus.fusion(args.model_name)
    elif args.mix_texture_depth:
        #generate S_{d+t}
        #args.model_name is S_d
        #args.texture_model_name is S_dt
        fus.fusion_texture_depth_mix_mode(args.model_name, args.texture_model_name)
    else:
        #S_dt
        fus.fusion_texture_depth(args.model_name)

