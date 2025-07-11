alphas=(0.3 0.5)
num_wms=(20)
steps=(0)


for alpha in ${alphas[@]};
do
    for num_wm in ${num_wms[@]};  
    do
        python -u train_DeepSign.py \
                --seed=15 \
                --alpha=${alpha} \
                --device_id=2 \
                --num_epochs=500 \
                --num_wm=${num_wm} \
                --seed=15 
    done
done