# choose setting, version=base/large/zero, datatrain=data_mix/data_mlm

version='base'
save='save_model/ckpt_decbasepre'
datatrain='predata/data_mix.jbl'
batch=8
#lm=''
lm='--pth_lm data/aug_decoder.txt'

#version='large' 
#save='save_model/ckpt_largemskpre'
#datatrain='predata/data_mlm.jbl'
#batch=2


# run the code
python mix_pretraining.py $lm --version $version --save_pth $save --pth_train $datatrain --batch_size $batch
