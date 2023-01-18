import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
from omegaconf import DictConfig
import onnxruntime
from stt_inferencer import _onnx_prediction
import numpy as np
import os
import json
import torch


lm_path = 'kenlm_3gram.arpa'
stt_config = "configs/quartznet15x5.yaml"
files = ["test.wav"]

yaml = YAML(typ='safe')
with open(stt_config, encoding="utf-8") as f:
    params = yaml.load(f)

beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
    vocab=list(params["labels"]),
    beam_width=10,
    alpha=2, beta=1.5,
    lm_path=lm_path,
    num_cpus=2,
    input_tensor=False)

# STT model prediction
prediction = self._onnx_prediction(files)
prediction = [[(-1.0, prediction[0])]]
print("stt prediction:", prediction)
# KENLM model prediction
logits = _onnx_prediction(files, return_logits_only=True)
probs = softmax(logits)
corrected_prediction = beam_search_lm.forward(log_probs=np.expand_dims(probs, axis=0), log_probs_length=None)
print("corrected prediction:", corrected_prediction)

