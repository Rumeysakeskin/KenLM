# KenLM

Utilized from [KenLM: Faster and Smaller Language Model Queries](https://github.com/kpu/kenlm) and it has more information about KenLM.

In this repository we will train language model on **Turkish corpus dataset** and deploy the model to inference after speech recognition for auto-correction.

### Train KenLM
- #### Compress a file using bzip2 command:
```
bzip2 corpus.txt
```

- #### Train language model by 3-gram:
Run the following command to create a `kenlm_3gram.arpa` model.
```
bzcat corpus.txt.bz2 | python preprocess.py | ./kenlm/bin/lmplz -o 3 > kenlm_3gram.arpa
```
### Inference KenLM  
We will use KenLM Model to the output of the STT inference for auto-correction.
```
python kenlm_inference.py
```
- #### Benchmark of Original STT output and KenLM output
```
expected output: naber bugün nasılsın 

stt prediction: naber bugun nasılsım
corrected prediction: naber bugün nasılsın
```

### Future Works

This STT utulized a pre-trained QuartzNet 15x5 which is a Character Encoding CTC Model.
Connectionist Temporal Classification loss function has a few limitations:

- Generated tokens are conditionally independent of each other. In other words - the probability of character "l" being predicted after "hel##" is conditionally independent of the previous token - so any other token can also be predicted unless the model has future information as we can see in the output of the stt prediction.

- The length of the generated (target) sequence must be shorter than that of the source sequence.

NVIDIA Nemo proposes **subword tokenization** since help alleviate both of these issues! 
You can find more information about subword tokenization in [ASR_with_Subword_Tokenization](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_Subword_Tokenization.ipynb).

Briefly, we can perform a larger number of pooling steps in our acoustic models, thereby improving execution speed while simultaneously reducing memory requirements.

Follow my [ASR-fine-tuning-for-low-resource-languages](https://github.com/Rumeysakeskin/ASR-fine-tuning-for-low-resource-languages) repository to train Sub-word Encoding CTC Model on Turkish dataset.

