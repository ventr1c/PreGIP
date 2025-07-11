alphas=(0.01 0.03 0.05 0.1 0.3 1.0)
watermark_sizes=(1)
num_wms=(20 40 100)
epss=(2.0)
steps=(3)


for alpha in ${alphas[@]};
do
for num_wm in ${num_wms[@]};  
do
    for step in ${steps[@]};
    do
        python -u run_random.py \
                --seed=15 \
                --alpha=${alpha} \
                --eps=2.0 \
                --step=${step} \
                --device_id=0 \
                --num_epochs=500 \
                --num_wm=${num_wm} \
                --seed=15
    done
done
done