job_name_wo_time='b5k_s1'

work_base='/anonymous/path'

current_time=$(date "+%m%d%H%M%S")
job_name="${current_time}_evaldet_${job_name_wo_time}"

scripts_dir="${work_base}/scripts"
results_dir="${work_base}/results/test/${job_name}"

cd "${work_base}"
mkdir -p $results_dir

bb='CLIP'
data_dir="${work_base}/eval/exp1/9_b5k_baseline/data/b5k_org_data"

nohup \
python "${backup_file_dir}/scripts/test_det.py" \
    --data_dir "${data_dir}" --result_dir ${results_dir} \
    --test_subj_idx 1 --test_epoch 5 \
    --num_ic 100 \
    --model_type "pretrained_ICL" --backbone_type $bb\
    --model_ckpt_path "ckpt.ckpt" \
    --gpu 4 \
    --batch_size 128 --save_all \
>> "${results_dir}/predict.out" 2>&1 &