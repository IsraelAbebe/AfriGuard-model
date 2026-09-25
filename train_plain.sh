# AfriGuard without the safety instruction or <safety>/<category>/<response> tags: user input -> response text.
# Expects HF_TOKEN to be set in the environment.

# Build data/afriguard_plain_{train,validation}.json from israel/AfriGuard-inst (only needed once).
[ -f data/afriguard_plain_train.json ] || python scripts/afriguard_plain.py

# Configs are named afriguard_plain_<base model>_<full|lora>_sft.yaml; each mirrors the AfriGuard-inst config in
# the comment, with only the dataset and output paths changed.
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain_afriqueqwen3.5-4b_full_sft.yaml  # afriguard2.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain_afriqueqwen3.5-4b-50langs_full_sft.yaml  # base of israel/AfriGuard-AfriqueQwen3.5-4B-50Langs
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain_tiny-aya-global_full_sft.yaml  # afriguard_tinyaya_full_sft.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/afriguard_plain_afriqueqwen3.5-4b-50langs-instruct_lora_sft.yaml  # afriguard_lora_sft.yaml
# FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/afriguard_plain_tiny-aya-global_lora_sft.yaml  # afriguard_tinyaya_lora_sft.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/afriguard_plain_afriqueqwen3.5-4b-50langs-instruct_full_sft.yaml  # afriguard.yaml
