#!/bin/bash

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-BertForMaskedLM_c-bert-base-uncased.json \
--output_file results/stereoset/results_stereoset_m-BertForMaskedLM_c-bert-base-uncased.json

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-ColakeForMaskedLM_c-colake.json \
--output_file results/stereoset/results_stereoset_m-ColakeForMaskedLM_c-colake.json

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-LukeForMaskedLM_c-luke-base.json \
--output_file results/stereoset/results_stereoset_m-LukeForMaskedLM_c-luke-base.json

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-RobertaForMaskedLM_c-roberta-base.json\
--output_file results/stereoset/results_stereoset_m-RobertaForMaskedLM_c-roberta-base.json

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-GPT2LMHeadModel_c-gpt2-m-kelm.json\
--output_file results/stereoset/results_stereoset_m-GPT2LMHeadModel_c-gpt2-m-kelm.json

python experiments/stereoset_evaluation.py --predictions_file \
results/stereoset/stereoset_m-GPT2LMHeadModel_c-gpt2-medium.json\
--output_file results/stereoset/results_stereoset_m-GPT2LMHeadModel_c-gpt2-medium.json