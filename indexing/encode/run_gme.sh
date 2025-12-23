for encoder_type in gme; do
    for dataset in M3DocVQA OpenDocVQA; do
        python encoder.py --encoder_type $encoder_type --dataset $dataset  &
    done
    wait
done


for encoder_type in gme; do
    for dataset in ViMDoc ViDoSeek; do
        python encoder.py --encoder_type $encoder_type --dataset $dataset &
    done
    wait
done
