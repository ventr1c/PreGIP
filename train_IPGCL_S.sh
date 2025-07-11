alphas=(0.3 1 3 10 30 100 300)
watermark_sizes=(1 5 10 20)
num_wms=(20)
epss=(2.0)
steps=(3)

for step in ${steps[@]};
do
    for alpha in ${alphas[@]};
    do
        python -u train_IPGCL_S.py \
                --seed=15 \
                --alpha=${alpha} \
                --eps=2.0 \
                --step=${step} \
                --device_id=3 \
                --num_epochs=1000 \
                --num_repeat=50 \
                --seed=15
        done
done