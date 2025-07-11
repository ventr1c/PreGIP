alphas=(0.1 0.3 1.0 3 10 30)
watermark_sizes=(1 5 20 50)
num_wms=(20)
epss=(2.0)
steps=(3)

for step in ${steps[@]};
do
    for alpha in ${alphas[@]};
    do
        for watermark_size in ${watermark_sizes[@]};  
        do

        python -u train_IPGCL2.py \
                --seed=15 \
                --alpha=${alpha} \
                --eps=2.0 \
                --step=${step} \
                --device_id=1 \
                --num_epochs=1000 \
                --watermark_size=${watermark_size} \
                --seed=15
        done
    done
done