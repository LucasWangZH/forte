# Author : Zhihao Wang
# Date : 22/10/2020

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

# pylint: disable=logging-fstring-interpolation
import logging
import os
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.types import DataRequest
from forte.models.ner import utils
from forte.models.ner.model_factory import BioBertBC5CDR
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention
from collections import  OrderedDict
import torch.nn as nn
from examples.bc5cdr.bc5cdr_trainer import HParams
from pytorch_pretrained_bert import BertConfig






logger = logging.getLogger(__name__)


class BC5CDRPredictor(FixedSizeBatchProcessor):
    """
       Note that to use :class:`CoNLLNERPredictor`, the :attr:`ontology` of
       :class:`Pipeline` must be an ontology that include
       ``ft.onto.base_ontology.Token`` and ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.hp = None
        self.resource = None
        self.device = None

        self.train_instances_cache = []

        self.batch_size = 1

    @staticmethod
    def _define_context() -> Type[Annotation]:
        return Sentence

    @staticmethod
    def _define_input_info() -> DataRequest:
        input_info: DataRequest = {
            Token: [],
            Sentence: [],
        }
        return input_info

    def initialize(self, resources: Resources, configs: Config):
        #TODO: ner debug "batcher" field is needed inside the config_data
        configs.add_hparam("batcher", configs.config_data)

        super().initialize(resources, configs)

        self.resource = resources
        self.config_model = configs.config_model
        self.config_data = configs.config_data
        self.batch_size = self.config_data.test_batch_size

        if resources.get("device"):
            self.device = resources.get("device")
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')

        self.hp = HParams(self.config_model.VOCAB_FILE)

        if "model" not in self.resource.keys():
            bert_config = BertConfig(self.config_model.BERT_CONFIG_FILE)
            tmp_d = torch.load(self.config_model.BERT_WEIGHTS, map_location=self.device)

            state_dict = OrderedDict()
            for i in list(tmp_d.keys())[:199]:
                x = i
                if i.find('bert') > -1:
                    x = '.'.join(i.split('.')[1:])
                state_dict[x] = tmp_d[i]

            self.model = BioBertBC5CDR(config=bert_config, bert_state_dict=state_dict,
                                       vocab_len=len(self.hp.VOCAB), device=self.device)
        self.model = resources.get("model")
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        utils.set_random_seed(self.config_model.random_seed)

    @torch.no_grad()
    def predict(self, data_batch: Dict[str, Dict[str, List[str]]]) \
            -> Dict[str, Dict[str, List[np.array]]]:
        tokens = data_batch["Token"]
        sentences = tokens["text"]
        sents = []
        x = []
        is_heads = []
        seqlens = []
        for sent in sentences:
            tmp = ["[CLS]"] + list(sent) + ["[SEP]"]
            tmpx = []
            heads = []
            for w in tmp:
                token = self.hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                is_head = [1] + [0] * (len(token) - 1)
                xx = self.hp.tokenizer.convert_tokens_to_ids(token)
                tmpx.extend(xx)
                heads.extend(is_head)
            sents.append(tmp)
            x.append(tmpx)
            is_heads.append(heads)
            seqlens.append(len(tmpx))
        max_len = np.array(seqlens).max()
        #pad x
        res = [np.array(sample + [0] * (max_len - len(sample))) for sample in x] # 0: <pad>
        res = np.vstack(res)
        x = torch.from_numpy(res).long()
        self.model.eval()
        # batch_data = self.get_batch_tensor(sents, device=self.device)
        _,_,yhat = self.model(x,torch.Tensor([1, 2, 3])) # just a dummy y value
        yhat = list(yhat.cpu().detach().numpy())

        pred: Dict = {"Token": {"ner": [], "tid": []}}
        for i in range(len(tokens["tid"])):
            y_hat = [hat for hd, hat in zip(is_heads[i], yhat[i]) if hd == 1]
            y_hat = y_hat[1:]
            tids = tokens["tid"][i]
            assert len(y_hat) == len(tids)+1

            ner_tags = []
            for j in range(len(tids)):
                ner_tags.append(self.hp.idx2tag[y_hat[j]])

            pred["Token"]["ner"].append(np.array(ner_tags))
            pred["Token"]["tid"].append(np.array(tids))

        return pred

    def load_model_checkpoint(self, model_path=None):
        p = model_path if model_path is not None \
            else self.config_model.model_path
        ckpt = torch.load(p, map_location=self.device)
        logger.info(f"Restoring NER model from {self.config_model.model_path}")
        self.model.load_state_dict(ckpt["model"])

    def pack(self, data_pack: DataPack,
             output_dict: Optional[Dict[str, Dict[str, List[str]]]] = None):
        """
        Write the prediction results back to datapack. by writing the predicted
        ner to the original tokens.
        """

        if output_dict is None:
            return

        current_entity_mention: Tuple[int, str] = (-1, "None")

        for i in range(len(output_dict["Token"]["tid"])):
            # an instance
            for j in range(len(output_dict["Token"]["tid"][i])):
                tid: int = output_dict["Token"]["tid"][i][j]  # type: ignore

                orig_token: Token = data_pack.get_entry(tid)  # type: ignore
                ner_tag: str = output_dict["Token"]["ner"][i][j]

                orig_token.ner = ner_tag

                token = orig_token
                token_ner = token.ner
                assert isinstance(token_ner, str)
                if token_ner[0] == "B":
                    current_entity_mention = (token.span.begin, token_ner[2:])
                elif token_ner[0] == "I":
                    continue
                elif token_ner[0] == "O":
                    continue

                elif token_ner[0] == "E":
                    if token_ner[2:] != current_entity_mention[1]:
                        continue

                    entity = EntityMention(data_pack,
                                           current_entity_mention[0],
                                           token.span.end)
                    entity.ner_type = current_entity_mention[1]
                elif token_ner[0] == "S":
                    current_entity_mention = (token.span.begin, token_ner[2:])
                    entity = EntityMention(data_pack, current_entity_mention[0],
                                           token.span.end)
                    entity.ner_type = current_entity_mention[1]

    # TODO: change this to manageable size
    @classmethod
    def default_configs(cls):
        r"""Default config for NER Predictor"""

        configs = super().default_configs()
        # TODO: Batcher in NER need to be update to use the sytem one.
        configs["batcher"] = {"batch_size": 10}

        more_configs = {
            "config_data": {
                "train_path": "",
                "val_path": "",
                "test_path": "",
                "num_epochs": 200,
                "batch_size_tokens": 512,
                "test_batch_size": 16,
                "max_char_length": 45,
                "num_char_pad": 2
            },
            "config_model": {
                "output_hidden_size": 128,
                "dropout_rate": 0.3,
                "word_emb": {
                    "dim": 100
                },
                "char_emb": {
                    "dim": 30,
                    "initializer": {
                        "type": "normal_"
                    }
                },
                "char_cnn_conv": {
                    "in_channels": 30,
                    "out_channels": 30,
                    "kernel_size": 3,
                    "padding": 2
                },
                "bilstm_sentence_encoder": {
                    "rnn_cell_fw": {
                        "input_size": 130,
                        "type": "LSTMCell",
                        "kwargs": {
                            "num_units": 128
                        }
                    },
                    "rnn_cell_share_config": "yes",
                    "output_layer_fw": {
                        "num_layers": 0
                    },
                    "output_layer_share_config": "yes"
                },
                "learning_rate": 0.01,
                "momentum": 0.9,
                "decay_interval": 1,
                "decay_rate": 0.05,
                "random_seed": 1234,
                "initializer": {
                    "type": "xavier_uniform_"
                },
                "model_path": "",
                "resource_dir": ""
            },
            "batcher": {
                "batch_size": 16
            }
        }

        configs.update(more_configs)
        return configs
