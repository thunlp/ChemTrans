cmd=''
# choose the setting, version=base/large/zero, init=None/pretrained ckpt

#version='base'
#save='save_model/ckpt_decbase_'
#batch=16

version='large'
#cmd=$cmd'--init_checkpoint save_model/ckpt_basemskpre980.pt'
save='save_model/ckpt_declarge_'
batch=4

# run the code
python dec_pt.py $cmd --version $version --save_pth $save --batch_size $batch
