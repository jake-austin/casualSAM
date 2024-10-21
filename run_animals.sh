for f in  boar elephant pexels_videos_1526909 pexels-adrien-jacta-5362421 pexels-cottonbro-6853904 pexels-zlatin-georgiev-7173031 piglets production_id_3765334 production_id_4812201 seal tiger turtle; do 
python train_slam.py --config experiments/davis/midasv3_finetune_learnshift.yaml --gpu 2 --track_name $f --dataset_name local --log_dir './experiment_logs_localdata' --localdata_path ~/data/animals/images/$f --image_sequence_stride 3;
find ./ | grep uncertainty_net.pth | xargs rm; find ./ | grep depth_net.pth | xargs rm;
done