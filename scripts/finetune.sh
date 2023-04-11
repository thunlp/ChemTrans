# choose the setting

version='base'
init=''
save='save_model/finetune_base.pt'
seed=1234
log='log/ftbase_'
batch=8
few=1
mutual=0

#version='large'
#init='--init_checkpoint save_model/ckpt_deczeropre980.pt'
#save='save_model/finetune_declargepre2.pt'
#seed=2345
#log='log/ftdeclargepre2_'
#batch=2

cmd=''
#cmd='--pth_train data/aug_ --pth_dev data/dev_ --pth_test data/test_'
# run the code
python tuning.py $init --mutual $mutual --few $few --version $version --save_pth $save --log_pth $log --seed $seed --batch_size $batch $cmd
