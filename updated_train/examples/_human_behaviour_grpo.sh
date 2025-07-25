#!/bin/bash
set -euo pipefail

# Optional: activate your virtual environment here if needed
# source /path/to/venv/bin/activate

# Train files is what you are training on
# Val files is what you are validating on
# Image directory is where the images are stored
# This does not cover testing

python -m verl.trainer.main \
    config=/home/keaneong/human-behavior/updated_train/examples/grpo_climb_engaging.yaml \
    data.train_files=/home/keaneong/human-behavior/data/instruction/meld_instruction_train.jsonl \
    data.val_files=/home/keaneong/human-behavior/data/instruction/meld_instruction_val.jsonl \
    data.image_dir=/scratch/keane/human_behaviour_data \
    algorithm.adv_estimator=grpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=1 \
    worker.actor.optim.lr=5e-6 \
    trainer.experiment_name=grpo_vanilla_unified

