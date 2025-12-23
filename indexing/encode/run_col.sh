for encoder_type in colpali colqwen2 colqwen25; do
    for dataset in M3DocVQA OpenDocVQA ViMDoc ViDoSeek; do
        python encoder.py --encoder_type $encoder_type --dataset $dataset &
    done
    wait
done

