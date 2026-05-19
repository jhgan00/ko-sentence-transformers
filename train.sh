# To start training, you need to download the KorNLUDatasets first.
# git clone https://github.com/kakaobrain/KorNLUDatasets.git

# train on STS dataset only
#python training_sts.py --model_name_or_path klue/bert-base
#python training_sts.py --model_name_or_path klue/roberta-base
#python training_sts.py --model_name_or_path klue/roberta-small
#python training_sts.py --model_name_or_path klue/roberta-large

# train on both NLI and STS dataset (multi-task)
#python training_multi-task.py --model_name_or_path klue/bert-base
python training_multi-task.py --model_name_or_path klue/roberta-base
#python training_multi-task.py --model_name_or_path klue/roberta-small
#python training_multi-task.py --model_name_or_path klue/roberta-large

# train on NLI dataset only
#python training_nli.py --model_name_or_path klue/bert-base
#python training_nli.py --model_name_or_path klue/roberta-base
#python training_nli.py --model_name_or_path klue/roberta-small
#python training_nli.py --model_name_or_path klue/roberta-large

# train with Matryoshka Representation Learning (MRL)
# Wrap the loss so the first m dims of the embedding are themselves usable
# at m in {768, 512, 256, 128, 64, 32}.
#python training_multi-task.py --model_name_or_path klue/roberta-base --matryoshka_dims 768,512,256,128,64,32
#python training_nli.py --model_name_or_path klue/roberta-base --matryoshka_dims 768,512,256,128,64,32
#python training_sts.py --model_name_or_path klue/roberta-base --matryoshka_dims 768,512,256,128,64,32