alphas=(1)
wm_nums=(10 30 50 100 200 500)
wm_node_nums=(1 5 30 50 100)

for alpha in ${alphas[@]};
do 
    for wm_num in ${wm_nums[@]};
    do
        for wm_node_num in ${wm_node_nums[@]};
        do  
            echo "alpha: $alpha, wm_num: $wm_num, wm_node_num: $wm_node_num"
            python run_random.py --dataset PROTEINS --alpha $alpha --num_wm $wm_num --wm_num_node $wm_node_num
        done
    done
done
