
export HUGGINGFACE_TOKEN=""
export HF_TOKEN=""


hf auth login --token "" --add-to-git-credential



# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard.yaml preprocessing_num_workers=2 dataloader_num_workers=0
   
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard.yaml 