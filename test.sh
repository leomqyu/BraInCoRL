job_name_wo_time='CLIP_s1'

work_base='.'

current_time=$(date "+%m%d%H%M%S")
job_name="${current_time}_${job_name_wo_time}"

scripts_dir="${work_base}/scripts"
results_dir="${work_base}/results/test/${job_name}"
ckpt_dir="${work_base}/checkpoints"

cd "${work_base}"
mkdir -p $results_dir

bb='CLIP' 
data_dir="${work_base}/data"

nohup \
python "${scripts_dir}/test_det.py" \
    --data_dir "${data_dir}" --result_dir ${results_dir} \
    --test_subj_idx 1 --test_epoch 5 \
    --num_ic 100 \
    --model_type "pretrained_ICL" --backbone_type $bb\
    --model_ckpt_path "${ckpt_dir}/CLIP_subj1.ckpt" \
    --gpu 0 \
    --batch_size 128 --save_all \
>> "${results_dir}/predict.out" 2>&1 &