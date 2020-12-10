# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict

from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from examples.bc5cdr.biobert_ner_predictor import BC5CDRPredictor
from examples.bc5cdr.bc5cdr_trainer import HParams
from ft.onto.base_ontology import Sentence, Token
import numpy as np

class BC5CDREvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.test_component = BC5CDRPredictor().name
        self.output_file = "tmp_eval.txt"
        self.hp = HParams(VOCAB_FILE=None)
        self.scores: Dict[str, float] = {}
        self.epoch = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
    def reset_res(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.scores = {}
    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack,epoch):
        if self.epoch != epoch:
            self.reset_res()
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["ner"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["ner"]},
                Sentence: [],  # span by default
            }
        }

        token_list,true_list,pred_list = get_label_lists(pred_pack=pred_pack,
                            pred_request=pred_getdata_args,
                            refer_pack=refer_pack,
                            refer_request=refer_getdata_args,
                            output_filename=self.output_file)
        y_true = np.array([self.hp.tag2idx[tmp] for tmp in true_list])
        y_pred = np.array([self.hp.tag2idx[tmp] for tmp in pred_list])

        self.TP += (np.logical_and(y_true == y_pred, y_true != 2)).astype(np.int).sum()
        self.TN += (np.logical_and(y_true == y_pred, y_true == 2)).astype(np.int).sum()
        self.FP += (np.logical_and(y_true != y_pred, y_pred != 2)).astype(np.int).sum()
        self.FN += (np.logical_and(y_true != y_pred, y_pred == 2)).astype(np.int).sum()

        all_negative_pred = self.TN + self.FN
        all_positive_pred = self.TP + self.FP
        all_positive_true = self.TP + self.FN
        all_crct = self.TP + self.TN

        #only considering fully correct
        def zero_division(n, d):
            return n / d if d else 0
        acc = zero_division(all_crct,(all_negative_pred+all_positive_pred))
        precision = zero_division(self.TP, all_positive_pred)
        recall = zero_division(self.TP,all_positive_true)
        f1 = zero_division((2 * precision * recall),(precision + recall))

        self.scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # for k,v in self.scores.items():
        #     if k == "accuracy":
        #         continue
        #     print(f"{k} is {v}")
        self.epoch =epoch

    def get_result(self):
        return self.scores

def get_label_lists(pred_pack, pred_request, refer_pack, refer_request,
                         output_filename):
    pred_list = []
    true_list = []
    token_list = []
    for pred_sentence, tgt_sentence in zip(
            pred_pack.get_data(**pred_request),
            refer_pack.get_data(**refer_request)
    ):

        pred_tokens, tgt_tokens = (
            pred_sentence["Token"],
            tgt_sentence["Token"],
        )
        for i in range(len(pred_tokens["text"])):
            w = tgt_tokens["text"][i]
            tgt = tgt_tokens["ner"][i]
            pred = pred_tokens["ner"][i]
            token_list.append(w)
            true_list.append(tgt)
            pred_list.append(pred)

    return token_list,true_list,pred_list
