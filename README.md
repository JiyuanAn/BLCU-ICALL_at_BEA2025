# BEA-2025-Shared-Task


## 一、数据预处理
### 1.1 将共享任务中提供的开发集划分为新的训练集和开发集。
```
python src/DataProcessing/dataset_partitioning.py
```

### 1.2 数据纠正与清洗
（请参见论文）

## 二、主要方法

### 2.1 基于上下文学习的方法
即将补充

### 2.2 基于监督微调的方法
2.2.1 数据格式处理
```
python ./src/SFT/Track1MistakeIdentification/data_formatting_track1_train.py
python ./src/SFT/Track1MistakeIdentification/data_formatting_track1_dev.py
python ./src/SFT/Track2MistakeLocation/data_formatting_track2_train.py
python ./src/SFT/Track2MistakeLocation/data_formatting_track2_dev.py
python ./src/SFT/Track3ProvidingGuidance/data_formatting_track3_train.py
python ./src/SFT/Track3ProvidingGuidance/data_formatting_track3_dev.py
python ./src/SFT/Track4Actionability/data_formatting_track4_train.py
python ./src/SFT/Track4Actionability/data_formatting_track4_dev.py
```

2.2.2 基于Lora的模型训练
```
llamafactory-cli train ./src/SFT/LLaMA-Factory/script/train_lora/lora_sft_track1.yaml
llamafactory-cli train ./src/SFT/LLaMA-Factory/script/train_lora/lora_sft_track2.yaml
llamafactory-cli train ./src/SFT/LLaMA-Factory/script/train_lora/lora_sft_track3.yaml
llamafactory-cli train ./src/SFT/LLaMA-Factory/script/train_lora/lora_sft_track4.yaml
```

2.2.3 Adapter模型合并
```
llamafactory-cli export ./src/SFT/LLaMA-Factory/script/merge_lora/merge_lora_sft_track1.yaml
llamafactory-cli export ./src/SFT/LLaMA-Factory/script/merge_lora/merge_lora_sft_track2.yaml
llamafactory-cli export ./src/SFT/LLaMA-Factory/script/merge_lora/merge_lora_sft_track3.yaml
llamafactory-cli export ./src/SFT/LLaMA-Factory/script/merge_lora/merge_lora_sft_track4.yaml
```

2.2.4 模型部署
建议基于vLLM框架部署后训练得到的模型。

2.2.5 效果评估


### 2.3 基于强化学习的方法
即将补充