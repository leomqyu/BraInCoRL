work_base='/anonymous/path'
job_name_wo_time='CLIP_pt_pmse_stg1_val_subj1'
current_time=$(date "+%m%d%H%M%S")
job_name="${current_time}_${job_name_wo_time}"

scripts_dir="${work_base}/scripts"
results_dir="${work_base}/results/train/${job_name}"


cd "${work_base}"
mkdir -p $results_dir

data_dir="${work_base}/data/results/fk_full_cor_pretrain_CLIP_gdgt"

backup_file_dir="${results_dir}/backup_files"
mkdir -p $backup_file_dir
cp -r ${scripts_dir} "${work_base}/1_train.sh" "$backup_file_dir/"
touch "${results_dir}/info.md"

nohup \
python "${backup_file_dir}/scripts/train_pl_mul_subj.py" \
    --work_base "${work_base}" --job_name "${job_name}" \
    --data_dir "${data_dir}" \
    --train_subj 0 --val_subj 0 \
    --default_struct CLIP \
    --pretrained_model_path "" --resume_ckpt_path "" \
    --min_in_context 30 --max_in_context 500 --max_unknown 100 \
    --gpu 4 \
    --batch_size 80 --lr 1e-4 --weight_decay 1e-1 --loss_type pmse --dropout 0.5 \
    --epochs 1000 \
>> "${results_dir}/predict.out" 2>&1 &