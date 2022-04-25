if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    SLURM_ARRAY_TASK_ID=0
fi

# Needed such that matplotlib does not produce error because XDG_RUNTIME_DIR not set
export MPLBACKEND=Agg
# Needed in newer versions of torch/cudnn
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

#Change to directory from which job was submitted
cd "$HOME/control-diffusion"

module load conda
source activate graph

gridentry=($(python scripts/paramgrid.py --id $SLURM_ARRAY_TASK_ID --sample 10))

python lib/run_tracing.py -a temp/agent_$SLURM_ARRAY_TASK_ID.json -r temp/amodel_$SLURM_ARRAY_TASK_ID.json \
    --nettype "barabasi:5:1" \
    --reindex False \
    --netsize 100 \
    --k 2.8 \
    --p .1 \
    --rem_orphans False \
    --use_weights True \
    --dual 0 \
    --control_schedule 2 0.5 0.99 \
    --control_after ${gridentry[1]} \
    --control_after_inf .05 \
    --control_initial_known ${gridentry[0]} \
    --control_gpu 0 \
    --first_inf 5 \
    --taut 0 \
    --taur 0 \
    --sampling_type "min" \
    --presample 10000 \
    --model "seir" \
    --spontan False \
    --pa=.2 \
    --update_after 1 \
    --summary_splits 20 \
    --noncomp 0 \
    --noncomp_after 10000 \
    --avg_without_earlystop True \
    --seed 21 \
    --netseed 25 \
    --infseed -1 \
    --multip 0 \
    --nnets 1 \
    --niters 1 > data/runs/run_stats.json
    
# Remove running artefacts from sim results
sed -i -n '/args/,$p' data/runs/run_stats.json