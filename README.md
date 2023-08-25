# ChemTrans

Source code for *[Transcription between Human-readable Synthetic Descriptions and Machine-executable Instructions: An Application of Latest Pre-training Technology](https://pubs.rsc.org/en/content/articlelanding/2023/SC/D3SC02483K)*. Our operating system is Ubuntu 16.04. For training process, the 3090 GPU is used.

## Simplified Instruction

If you want to quickly explore our job or do not have much deep-learning experience, you can simply follow the instructions in this section.

- Step 1: Download the zip or clone the repository to your workspace.
- Step 2: Download the `D2IPTM-base/large.pt` and `I2DPTM-base/large.pt` from [googledrive](https://drive.google.com/drive/folders/1AT-1uUR5Ev5d8fxX2FFrIZSnQ1-jbiG6?usp=sharing). Create a new directory by `mkdir save_model` and then put the downloaded model under `save_model/` directory.
- Step 3: Install Anaconda (py3) and then create a conda environment by the following command (remember to input 'y' when asked `Proceed ([y]/n)? `):
```
conda create -n ChemTrans python=3.9
conda activate ChemTrans
sh scripts/conda_environment.sh
```
  Note that there may be error when installing transformers if you're using MacOS. See [here](https://github.com/huggingface/transformers/issues/2831) for help.
- Step 4: Check the `interact.py` file and set `if_cuda=False` in line 25 if there is no GPU available. Run the command:
```
python interact.py
```
  And then explore the ChemTrans task following the instructions of the program:
~~~bash
# input the synthesis natural language description and then type enter
SI input: >> an oven-dried Schlenk flask equipped with a magnetic stir bar was charged with N-bromosuccinimide (1.26 g, 7.1 mmol, 1.25 equiv.) and sealed. The flask was evacuated and backfilled with nitrogen (this sequence was repeated a total of three times). To the flask were added a solution of S11 (0.89 g, 5.7 mmol, 1.0 equiv.) in acetone (35 mL) via syringe. The flask was opened and silver nitrate (0.10 g, 0.6 mmol, 10 mol%) was added quickly. The flask was re-sealed, and the reaction mixture was stirred at r.t. for 2 h. The solvent was removed in vacuo and the residue was diluted with a mixture of PE:EA (v/v = 1:1). The resulting slurry was filtered through a plug of silica gel and eluted with PE:EA (v/v = 1:1). The filtrate was washed with water and brine, dried over anhydrous Na2SO4 and concentrated in vacuo. The residue was purified by silica gel column chromatography (PE:EA= 40:1 – 4:1) to afford S12 (1.17 g, 88%) as a pink solid.
# the program will return the predicted instructions
[ add ] reagent: ( name: N-bromosuccinimide & type: pure & mole: 7.1 mmol & mass: 1.26 g & concentration: 1.25 equiv & ) & [ add ] reagent: ( name: S11 & mole: 5.7 mmol & mass: 0.89 g & equivalent: 1.0 equiv & ) & reagent: ( name: acetone & type: pure & volume: 35 mL & ) & reagent: ( name: a solution of S11 0.89 g, 5.7 mmol, 1.0 equiv. in acetone 35 mL & type: mixture & ) & [ add ] reagent: ( name: silver nitrate & type: pure & mole: 0.6 mmol & mass: 0.10 g & concentration: 10 mol% & speed: quickly. & ) & [ settemp ] time: 2 h. & [ evaporate ] N/A: removed & [ add ] reagent: ( name: PE:EA & type: mixture & ) & [ filter ] reagent: ( name: silica gel & type: pure & ) & [ column ] reagent(adsorbent): silica gel & [ wash ] reagent: ( name: water & type: pure & ) & reagent: ( name: brine & type: pure & ) & [ dry ] reagent: ( name: Na2SO4 & type: pure & note: anhydrous & ) & [ evaporate ] N/A: concentrated & [ column ] reagent(eluent): PE:EA= 40:1 – 4:1 & [ yield ] appearance: a pink solid & yield: 88% & mass(yield): 1.17 g &

# automatically loop until you type control+C
SI input: >>
~~~
  Or provide input instructions with the help of the program, and get the transcribed descriptions. Remember to set `if_mutual=1` in line 26.
~~~bash
# input the corresponding number for the operations and augments
{1: 'add', 2: 'settemp', 3: 'yield', 4: 'wash', 5: 'filter', 6: 'evaporate', 7: 'dry', 8: 'distill', 9: 'extract', 10: 'transfer', 11: 'reflux', 12: 'recrystallize', 13: 'quench', 14: 'column', 15: 'triturate', 16: 'partition'}
choose operation (17 for stop): >> 1
{1: 'temperature', 2: 'reagent'}
choose augments (0 for stop): >> 2
{1: 'name', 2: 'type', 3: 'mass', 4: 'volume', 5: 'speed', 6: 'concentration', 7: 'equivalent', 8: 'batch', 9: 'note', 10: 'temperature', 11: 'mole'}
choose reagent augments (0 for stop): >> 1
# input the value for the augments
value for name: >> N-bromosuccinimide
choose reagent augments (0 for stop): >> 2
value for type: >> pure
choose reagent augments (0 for stop): >> 3
value for mass: >> 1.26 g
choose reagent augments (0 for stop): >> 11
value for mole: >> 7.1 mmol
choose reagent augments (0 for stop): >> 6
value for concentration: >> 1.25 equiv
choose reagent augments (0 for stop): >> 0
choose augments (0 for stop): >> 0
choose operation (17 for stop): >> 2
{1: 'time', 2: 'temperature'}
choose augments (0 for stop): >> 1
value for time: >> 2 h.
choose augments (0 for stop): >> 0
choose operation (17 for stop): >> 6
{1: 'temperature', 2: 'pressure'}
choose augments (0 for stop): >> 1
value for temperature: >> removed
choose augments (0 for stop): >> 0
choose operation (17 for stop): >> 5
{1: 'reagent', 2: 'phase'}
choose augments (0 for stop): >> 1
{1: 'name', 2: 'type', 3: 'mass', 4: 'volume', 5: 'speed', 6: 'concentration', 7: 'equivalent', 8: 'batch', 9: 'note', 10: 'temperature', 11: 'mole'}
choose reagent augments (0 for stop): >> 1
value for name: >> silica gel
choose reagent augments (0 for stop): >> 2
value for type: >> pure
choose reagent augments (0 for stop): >> 0
choose augments (0 for stop): >> 0
choose operation (17 for stop): >> 3
{1: 'reagent name', 2: 'appearance', 3: 'mass(yield)', 4: 'yield'}
choose augments (0 for stop): >> 2
value for appearance: >> a pink solid
choose augments (0 for stop): >> 4
value for yield: >> 88%
choose augments (0 for stop): 3
value for mass(yield): >> 1.17 g
choose augments (0 for stop): 0
choose operation (17 for stop): 17
# check the input, modify and ensure your instructions
Your input instructions are:  [ add ] reagent: ( name: N-bromosuccinimide & type: pure & mass: 1.26 g & mole: 7.1 mmol & concentration: 1.25 equiv & )  [ settemp ] time: 2 h. &  [ evaporate ] temperature: removed &  [ filter ] reagent: ( name: silica gel & type: pure & )  [ yield ] appearance: a pink solid & yield: 88% & mass(yield): 1.17 g & 
verify your input: [ add ] reagent: ( name: N-bromosuccinimide & type: pure & mass: 1.26 g & mole: 7.1 mmol & concentration: 1.25 equiv & )  [ settemp ] time: 2 h. &  [ evaporate ] N/A: removed &  [ filter ] reagent: ( name: silica gel & type: pure & )  [ yield ] appearance: a pink solid & yield: 88% & mass(yield): 1.17 g &       
# get the transcribed result
To a solution of tert-butyl 2-(dimethylamino)cyclohexane-1,3-dione (1.17 g, 7.1 mmol, 88%) in tetrahydrofuran (50 mL) is added N-bromosuccinimide (1.26 g, 7.1 mmol, 1.25 equiv). The flask is fitted with a magnetic stirring bar and the solution is stirred under nitrogen for 2 h. The solvent is removed with a rotary evaporator and the residue is filtered through a short pad of silica gel (elution with ethyl acetate-hexanes) to afford the product as a pink solid (1.17 g, 88%).
~~~

## Requirements

We strongly suggest you to create a conda environment for this project. Installation is going to be finished in a several minutes.

```
conda create -n ChemTrans python=3.9
conda activate ChemTrans
sh scripts/conda_environment.sh
```

## Download

D2I/I2DPTM and other pre-trained models can be downloaded from [googledrive]([TODO](https://drive.google.com/drive/folders/1AT-1uUR5Ev5d8fxX2FFrIZSnQ1-jbiG6?usp=sharing)). We recommend you to download the models and put them under save\_model/ before running the code.

Pre-training corpus and the pre-processed data can also be downloaded from the above link. Please create a file called predata/ and put the data under it.

## File Usage

The users may be going to use the files below:

- mix\_pretraining: D2I knowledge-enhanced training code for all of the tasks.
- dis\_pretraining: I2D knowledge-enhanced training code for tasks except augmentation.
- dec\_pt.py: Training code for the decoder language modeling.
- tuning.py: Fine-tuning code for ChemTrans task.
- interact.py: Demo code for ChemTrans interaction.
- evalchem.py: Evaluation function code.
- D2I\_deval.py: Evaluation tool code.
- I2D\_deval.py: Evaluation tool code.
- gen\_aug.py: Decoder language modeling data augmentation generation code.
- LLM\_test.py: Large language model testing code.
- data/
  - token\_save.pkl: Special tokens for expanding T5 tokenizer.
  - \*\_inp.txt: Input text for train/dev/test/aug set.
  - \*\_out.txt: Label text for train/dev/test/aug set.
  - aug\_decoder.txt: Augmented text for decoder language modeling.
  - query\_instances.txt: The most similar training instances used for LLM testing.
- scripts/
  - conda\_environment.sh: Conda environment creation bash file.
  - decoder\_lm.sh: Decoder language modeling bash file.
  - finetune.sh: Fine-tuning bash file.
  - pretrain.sh: Knowledge-enhanced training bash file.

## Running the Code

We strongly recommend you to test our code with a GPU. Usually the downstream fine-tuning process takes no more than an hour (for base ver.).

For the **D2I training** period, we modify the `scripts/pretrain.sh` file according to our settings:

```
version='base'
# choose 'base', 'large' or 'zero' for the model scale, and 'zero' stands for the small Vanilla Transformer setting. Other model scale can be defined in the mix_pretraining.py file.
save='save_model/ckpt_decbasepre'
# path and name for model saving.
datatrain='predata/data_mix.jbl'
# data_mix.jbl for multi-task training, and data_mlm.jbl for purely masked language modeling.
batch=8
lm='--pth_lm data/aug_train.txt'
# set lm='' for post-training without decoder language modeling.
```
And run the bash file.
```
CUDA_VISIBLE_DEVICES=0 sh scripts/pretrain.sh
```
For **I2D training** period, replace the python file with `dis_pretraining.py`, and then use tuning code before fine-tuning, while replace the training text with `aug_inp/out.txt`. 

For the **fine-tuning** period, we modify the `scripts/finetune.sh` file according to our settings:
```
mutual=0
# set 0 for D2I and 1 for I2D
version='base'
# choose 'base', 'large' or 'zero' for the model scale
init='--init_checkpoint save_model/ckpt_basepre980.pt'
# set init='' if there is no need for initializing the T5 model
save='save_model/finetune_basepre.pt'
# path and name for model saving.
log='log/ftbasepre_'
# path and name for the predicted context.
few=1
# set the data ratio used for training
```
And run the bash file.
```
CUDA_VISIBLE_DEVICES=0 sh scripts/finetune.sh
```

For the evaluation, run the code according to the log file that the model generated:
```
python D2I_deval.py LOG_FILE(e.g. log/ftbasepre_)
python I2D_deval.py LOG_FILE(e.g. log/ftrevbasepre_)
```

## Citation
Please cite our paper if you find it helpful.
```
@Article{D3SC02483K,
author ="Zeng, Zheni and Nie, Yi-Chen and Ding, Ning and Ding, Qian-Jun and Ye, Wei-Ting and Yang, Cheng and Sun, Maosong and E, Weinan and Zhu, Rong and Liu, Zhiyuan",
title  ="Transcription between human-readable synthetic descriptions and machine-executable instructions: an application of the latest pre-training technology",
journal  ="Chem. Sci.",
year  ="2023",
pages  ="-",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/D3SC02483K",
url  ="http://dx.doi.org/10.1039/D3SC02483K"
}
```
