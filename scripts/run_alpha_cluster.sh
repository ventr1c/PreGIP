alphas=(0.001 0.01 0.1 1 10 100)
# wm_nums=(10 30)
# wm_node_nums=(1 5 30)
wm_nums=(20 50 100)
watermark_sizes=(1 5 30)
for alpha in ${alphas[@]};
do 
    for wm_num in ${wm_nums[@]};
    do
        for watermark_size in ${watermark_sizes[@]};
        do  
            echo "alpha: $alpha, wm_num: $wm_num, watermark_size: $watermark_size"
            python -u run_random_select.py --dataset PROTEINS --normal_select cluster --alpha $alpha --num_wm $wm_num --watermark_size $watermark_size --device 1
        done
    done
done
