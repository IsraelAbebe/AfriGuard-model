# AfriGuard without the safety instruction or <safety>/<category>/<response> tags: user input -> response text.
# Expects HF_TOKEN to be set in the environment.

# Build data/afriguard_plain_{train,validation}.json from israel/AfriGuard-inst (only needed once).
[ -f data/afriguard_plain_train.json ] || python scripts/afriguard_plain.py

# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard2_plain.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain_tinyaya_full_sft.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/afriguard_plain_lora_sft.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/afriguard_plain_tinyaya_lora_sft.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain.yaml
