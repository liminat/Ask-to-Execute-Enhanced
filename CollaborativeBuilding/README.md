# Enhanced Collaborative Building Task

## 1. Preprocessing Steps
```
cd builder
python data_loader_with_glove.py --split train
python data_loader_with_glove.py --split val
python data_loader_with_glove.py --split test
python dataloader_with_glove.py --split train --json_data_dir ../builder_data_with_glove
python dataloader_with_glove.py --split val --json_data_dir ../builder_data_with_glove
python dataloader_with_glove.py --split test --json_data_dir ../builder_data_with_glove
cd ..
```

## 2. Model Training
```
python train.py --json_data_dir builder_data_with_glove --saved_models_path saved_models_1
```

## 3. Testing
```
python test.py --json_data_dir data_path --saved_models_path model_path
```

The repository is now maintained by liminat. Originally developed by a developer (whose name must not be disclosed as per policy).

Github reference: [1]: https://github.com/prashant-jayan21/minecraft-bap-models