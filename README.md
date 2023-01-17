# KenLM

Utilized from [KenLM: Faster and Smaller Language Model Queries](https://github.com/kpu/kenlm) and it has more information about KenLM.

In this repository we will train language model on **Turkish corpus dataset** and deploy the model to inference after speech recognition for auto-correction.

### Train KenLM
- Compress a file using bzip2 command:
```
bzip2 corpus.txt
```

- Train language model by 3-gram:
```
bzcat corpus.txt.bz2 | python preprocess.py | ./kenlm/bin/lmplz -o 3 > kenlm_3gram.arpa
```
