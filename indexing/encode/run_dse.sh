for encoder_type in dse gme visret; do
    for dataset in ViMDoc ViDoSeek M3DocVQA OpenDocVQA; do
        python encoder.py --encoder_type $encoder_type --dataset $dataset &
    done
    wait
done
