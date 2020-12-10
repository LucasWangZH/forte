# Author : Zhihao Wang
# Date : 29/10/2020

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
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Dict
from collections import OrderedDict
import numpy as np
import torch
from torch.optim import Adam
import torchtext
from tqdm import tqdm

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.models.ner import utils
from forte.models.ner.model_factory import BioBertBC5CDR
from forte.trainer.base.base_trainer import BaseTrainer
from ft.onto.base_ontology import Token, Sentence
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertConfig
from torch.utils.data import Dataset,DataLoader
logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    '''
    focal loss from paper: https://arxiv.org/pdf/1708.02002v2.pdf
    '''
    def __init__(self, gamma=2,alpha=0.25,scale=10,reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.scale = scale
    def forward(self, input, target):
        one_hot_target = torch.zeros(input.shape)
        one_hot_target[torch.arange(len(input)), target] = 1
        one_hot_target = one_hot_target.cuda()
        # one_hot_target = F.one_hot(labels, num_classes=input.shape[1])

        logprob = torch.nn.functional.log_softmax(input,dim=1)
        prob = torch.exp(logprob).cuda()
        prob_tmp_for_gamma = (1-prob) * one_hot_target + prob * (1-one_hot_target)
        prob_tmp_for_log = prob * one_hot_target + (1-prob) * (1 - one_hot_target)
        coefficient = -(self.alpha * one_hot_target + (1-self.alpha)* (1-one_hot_target))
        loss = coefficient * torch.pow(prob_tmp_for_gamma,self.gamma) * torch.log(prob_tmp_for_log)
        if self.reduction == "mean":
            return loss.mean()*self.scale
        else:
            return loss.sum()

class HParams:
    def __init__(self,VOCAB_FILE):
        self.VOCAB = ('<PAD>', 'B-Chemical', 'O', 'B-Disease' , 'I-Disease', 'I-Chemical')

        self.tag2idx = {v:k for k,v in enumerate(self.VOCAB)}
        self.idx2tag = {k:v for k,v in enumerate(self.VOCAB)}
        if VOCAB_FILE is not None:
            self.tokenizer = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

class BC5CDR_dataset(Dataset):
    def __init__(self,data):
        super(BC5CDR_dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def _pad(batch):
    '''Pads to the longest sample, and return tensor'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x),is_heads, tags, f(y), seqlens


class BC5CDRTrainer(BaseTrainer):
    def __init__(self):
        """ Create an NER trainer.
        """

        super().__init__()

        self.model = None

        self.config_model = None
        self.config_data = None
        self.normalize_func = None

        self.hp = None
        self.bert_config = None
        self.device = None
        self.optim, self.trained_epochs = None, None
        self.criterion = None
        self.resource: Optional[Resources] = None

        self.train_instances_cache = []

        # Just for recording
        self.max_char_length = 0

        self.__past_dev_result = None

    def initialize(self, resources: Resources, configs: Config):
        """
        The training pipeline will run this initialization method during
        the initialization phase and send resources in as parameters.

        Args:
            resources: The resources shared in the pipeline.
            configs: configuration object for this trainer.

        Returns:

        """
        self.resource = resources

        self.config_model = configs.config_model
        self.config_data = configs.config_data

        self.normalize_func = utils.normalize_digit_word

        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        utils.set_random_seed(self.config_model.random_seed)

        self.hp = HParams(self.config_model.VOCAB_FILE)
        self.bert_config = BertConfig(self.config_model.BERT_CONFIG_FILE)
        tmp_d = torch.load(self.config_model.BERT_WEIGHTS, map_location=self.device)

        state_dict = OrderedDict()
        for i in list(tmp_d.keys())[:199]:
            x = i
            if i.find('bert') > -1:
                x = '.'.join(i.split('.')[1:])
            state_dict[x] = tmp_d[i]

        self.model = BioBertBC5CDR(config=self.bert_config,bert_state_dict=state_dict,
                                   vocab_len=len(self.hp.VOCAB),device=self.device)
        # self.model.load_state_dict(torch.load("E:/CMU/Directed Study/Forte/forte/examples/bc5cdr/resources/2.pt"))
        if torch.cuda.is_available():
            self.model.cuda()

        self.optim = Adam([{'params':self.model.parameters()}], lr=self.config_model.learning_rate,
                          weight_decay=self.config_model.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        # self.criterion = FocalLoss(reduction="mean")
        if torch.cuda.is_available():
            self.criterion.cuda()

        self.trained_epochs = 0

        self.resource.update(model=self.model)

    def data_request(self) -> Dict:
        """
        Build a request to a :class:`DataPack <forte.data.data_pack.DataPack>`.
        The NER trainer will request the ner filed of the tokens, organized
        by each sentence.

        Returns:

        """

        request_string = {
            "context_type": Sentence,
            "request": {
                Token: ["ner"],
                Sentence: [],  # span by default
            }
        }
        return request_string

    def consume(self, instance):
        """
        Consumes the next NER instance for training. The instance will be
        collected until a batch is full.

        Args:
            instance: An instance contains a sentence, in the form of tokens and
                their "ner" tag.

        Returns:

        """

        tokens = instance["Token"]
        ners = tokens["ner"]

        word_ids = []
        ner_ids = []
        is_heads = []
        sentence = tokens["text"]
        words = ["[CLS]"] + list(sentence) + ["[SEP]"]
        ner_tags = ['<PAD>'] + list(ners) + ['<PAD>']

        for i,word in enumerate(words):
            token = self.hp.tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            xx = self.hp.tokenizer.convert_tokens_to_ids(token)
            word_ids.extend(xx)
            is_head = [1] + [0] * (len(token) - 1)
            t = [ner_tags[i]] + ["<PAD>"] * (len(token) - 1)  # <PAD>: no decision
            yy = [self.hp.tag2idx[each] for each in t]
            ner_ids.extend(yy)
            is_heads.extend(is_head)
        seq_len = len(ner_ids)
        # pad x

        self.train_instances_cache.append((words, word_ids,is_heads, ner_tags, ner_ids,seq_len))

    def epoch_finish_action(self, epoch):
        """
        At the end of each dataset_iteration, we perform the training,
        and set validation flags.

        :return:
        """
        counter = len(self.train_instances_cache)
        logger.info(f"Total number of ner_data: {counter}")

        lengths = \
            sum([len(instance[0]) for instance in self.train_instances_cache])

        logger.info(f"Average sentence length: {(lengths / counter):0.3f}")

        train_crct = 0.0
        train_total = 0.0
        train_loss = 0.0
        start_time = time.time()
        self.model.train()

        # Each time we will clear and reload the train_instances_cache
        instances = self.train_instances_cache
        # random.shuffle(self.train_instances_cache)
        dataset = BC5CDR_dataset(instances)
        train_loader = DataLoader(dataset=dataset,shuffle=True,
                                  num_workers=8,batch_size=self.config_data.batch_size,
                                  collate_fn=_pad,pin_memory=True)

        step = 0

        for batch in train_loader:
            step += 1

            words, x,is_heads, ner_tags, y,seq_len = batch

            _y = y  # for monitoring
            self.optim.zero_grad()
            logits, y, _ = self.model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)

            loss = self.criterion(logits, y)
            loss.backward()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            tmp = (predicted == y).squeeze().sum()
            tmp = tmp.cpu().detach()
            train_crct += tmp
            train_loss += loss.item()

            self.optim.step()

            # update log
            if step % 100 == 0:
                print(f"Train: {step}, "
                    # f"training acc: {(train_crct / train_total):0.3f}"
                      f"training loss {(train_loss / step):.3f}")

        logger.info(f"Epoch: {epoch}, steps: {step}, "
                    f"acc: {(train_crct / train_total):0.3f}, "
                    f"loss: {(train_loss / step):0.3f},"
                    f"time: {(time.time() - start_time):0.3f}s")

        self.trained_epochs = epoch

        # if epoch % self.config_model.decay_interval == 0:
        #     lr = self.config_model.learning_rate / \
        #          (1.0 + self.trained_epochs * self.config_model.decay_rate)
        #     for param_group in self.optim.param_groups:
        #         param_group["lr"] = lr
        #     logger.info(f"Update learning rate to {lr:0.3f}")

        self.request_eval()
        self.train_instances_cache.clear()

        if epoch >= self.config_data.num_epochs:
            self.request_stop_train()

    @torch.no_grad()
    def get_loss(self, instances: Iterator) -> float:
        """
        Compute the loss based on the validation data.

        Args:
            instances:

        Returns:

        """
        losses = 0
        val_dataset = BC5CDR_dataset(list(instances))
        val_loader = DataLoader(dataset=val_dataset,batch_size=self.config_data.test_batch_size)
        self.model.eval()
        for i,batch in enumerate(val_loader):
            words, x,is_heads, ner_tags, y,seq_len = batch
            logits, y, _ = self.model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = self.criterion(logits, y)
            losses += loss.item()

        mean_loss = losses / len(val_dataset)
        return mean_loss

    def post_validation_action(self, eval_result):
        """
        Log the evaluation results.

        Args:
            eval_result: The evaluation results.

        Returns:

        """

        if eval_result["eval"].get("accuracy") is None:
            return

        if self.__past_dev_result is None or \
                (eval_result["eval"]["f1"] >
                 self.__past_dev_result["eval"]["f1"]):
            self.__past_dev_result = eval_result
            logger.info("Validation f1 increased, saving model")
            self.__save_model_checkpoint()

        best_epoch = self.__past_dev_result["epoch"]
        acc, prec, rec, f1 = (self.__past_dev_result["eval"]["accuracy"],
                              self.__past_dev_result["eval"]["precision"],
                              self.__past_dev_result["eval"]["recall"],
                              self.__past_dev_result["eval"]["f1"])
        logger.info(f"Best val acc: {acc: 0.3f}, precision: {prec:0.3f}, "
                    f"recall: {rec:0.3f}, F1: {f1:0.3f}, epoch={best_epoch}")
        print(f"Best val acc: {acc: 0.3f}, precision: {prec:0.3f}, "
                    f"recall: {rec:0.3f}, F1: {f1:0.3f}, epoch={best_epoch}")

        if "test" in self.__past_dev_result:
            acc, prec, rec, f1 = (self.__past_dev_result["test"]["accuracy"],
                                  self.__past_dev_result["test"]["precision"],
                                  self.__past_dev_result["test"]["recall"],
                                  self.__past_dev_result["test"]["f1"])
            logger.info(f"Best test acc: {acc: 0.3f}, precision: {prec: 0.3f}, "
                        f"recall: {rec: 0.3f}, F1: {f1: 0.3f}, "
                        f"epoch={best_epoch}")


    def finish(self, resources: Resources):  # pylint: disable=unused-argument
        """
        Releasing resources and saving models.

        Args:
            resources: The resources used by the training process.

        Returns:

        """
        if self.resource:
            keys_to_serializers = {}
            for key in resources.keys():
                if key == "model":
                    keys_to_serializers[key] = \
                        lambda x, y: pickle.dump(x.state_dict(), open(y, "wb"))
                else:
                    keys_to_serializers[key] = \
                        lambda x, y: pickle.dump(x, open(y, "wb"))

            self.resource.save(keys_to_serializers,
                               output_dir=self.config_model.resource_dir)

        self.__save_model_checkpoint()

    def __save_model_checkpoint(self):
        states = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

        path = Path(self.config_model.model_path)
        if not Path(self.config_model.model_path).exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="wb") as f:
            torch.save(states, f)

    def load_model_checkpoint(self):
        """
        Load the model with a check pointer.

        Returns:

        """
        ckpt = torch.load(self.config_model.model_path)
        logger.info("restoring model from %s",
                    self.config_model.model_path)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])


