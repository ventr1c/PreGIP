alphas=(0.01 0.03 0.05 0.1 0.3 0.5 1.0 3)
watermark_sizes=(1)
num_wms=(20)
epss=(2.0)
steps=(50)


for num_wm in ${num_wms[@]}; 
do
    for watermark_size in ${watermark_sizes[@]};
    do
        for alpha in ${alphas[@]};
        do
            python -u run_random.py \
                    --seed=15 \
                    --alpha=${alpha} \
                    --eps=2.0 \
                    --step=10 \
                    --device_id=2 \
                    --num_epochs=500 \
                    --num_wm=${num_wm} \
                    --watermark_size=${watermark_size}
        done
    done
done