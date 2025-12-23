for encoder_type in visret; do
    for dataset in ViMDoc ViDoSeek M3DocVQA OpenDocVQA; do
        python encoder.py --encoder_type $encoder_type --dataset $dataset 
    done
done
wait