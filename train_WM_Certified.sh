alphas=(0.001 0.01 0.1 1.0)
watermark_sizes=(1 5 30)
num_wms=(20)
epss=(0.1 0.2 0.5 1.0 2.0)
steps=(3)


for alpha in ${alphas[@]};
do

    for eps in ${epss[@]};
    do
        for watermark_size in ${watermark_sizes[@]};  
        do
        python -u train_WM.py \
                --seed=15 \
                --alpha=${alpha} \
                --eps=${eps} \
                --step=1 \
                --random\
                --device_id=1 \
                --num_epochs=500 \
                --watermark_size=${watermark_size}\
                --seed=15
        done
    done
done