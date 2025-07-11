alphas=(0.001 0.01 0.1 1.0 10 100)
watermark_sizes=(3 5 10 20 30)
steps=(0)


for alpha in ${alphas[@]};
do
    for watermark_size in ${watermark_sizes[@]};  
    do
            python -u train_WPE.py \
                    --seed=15 \
                    --alpha=${alpha} \
                    --device_id=1 \
                    --num_epochs=500 \
                    --watermark_size=${watermark_size} \
                    --seed=15
    done
done