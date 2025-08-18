#!/bin/bash
# Define other parameters
data_path='./data/sudoku-extreme-1k-aug-1000'
dataset_name='sudoku'
epochs=20000
eval_interval=2000
lr=0.0001
puzzle_emb_lr=0.0001
weight_decay=1
puzzle_emb_weight_decay=1
start_seed=50
num_runs=1
render_res=288
output_size=224
global_batch_size=384



# Loop over all combinations
for seed in $(seq $start_seed $((start_seed + num_runs - 1))); do
    # Create a unique experiment ID
    exp_id="ix-s-${seed}-"
    echo "Submitting job for combination: $exp_id"
    
    # Create a temporary script file
    tmp_script=$(mktemp /home/pbhat1/projects/NeurAI/HRM/scripts/slurm_script.XXXXXX)
    cat <<EOF > "$tmp_script"
#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /home/pbhat1/projects/NeurAI/HRM/slurm/${dataset}/${seed}/slurm-%j.out
#SBATCH -e /home/pbhat1/projectsNeurAI/HRM/slurm/${dataset}/${seed}/slurm-%j.err

# Load necessary modules (adjust based on your environment)
source ~/miniconda3/bin/activate
conda activate hrm

export MASTER_PORT=\$((10000 + \$(echo -n \$SLURM_JOBID | tail -c 4)))
export WORLD_SIZE=\$((\$SLURM_NNODES * \$SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=\$WORLD_SIZE"

master_addr=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=\$master_addr
echo "MASTER_ADDR=\$MASTER_ADDR"

# Run the Python script with the current parameters
srun OMP_NUM_THREADS=96 torchrun --nproc-per-node 1 python /home/pbhat1/projects/NeurAI/HRM/pretrain.py \
  --data_path $data_path \
  --epochs $epochs \
  --eval_interval $eval_interval \
  --lr $lr \
  --puzzle_emb_lr $puzzle_emb_lr \
  --weight_decay $weight_decay \
  --puzzle_emb_weight_decay $puzzle_emb_weight_decay \
  --dataset_name $dataset_name \
  --render_res $render_res \
  --output_size $output_size \
  --global_batch_size $global_batch_size \
  
EOF
      
    # Submit the temporary script
    sbatch "$tmp_script"
    sleep 30
    rm "$tmp_script"
done