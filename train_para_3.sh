alphas=(1)
watermark_sizes=(1)
num_wms=(20)
epss=(0.1 0.5 1.0 2.0)
steps=(1 2 5 10)


for alpha in ${alphas[@]};
do
for eps in ${epss[@]};  
do
    for step in ${steps[@]};
    do
        python -u run_random.py \
                --seed=15 \
                --alpha=${alpha} \
                --eps=${eps} \
                --step=${step} \
                --device_id=2 \
                --num_epochs=500 \
                --num_wm=20
    done
done
done