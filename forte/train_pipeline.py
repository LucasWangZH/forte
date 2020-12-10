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

import logging
from typing import Optional, List

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.readers.base_reader import BaseReader
from forte.evaluation.base.base_evaluator import Evaluator
from forte.pipeline import Pipeline
from forte.processors.base import BaseProcessor
from forte.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self, train_reader: BaseReader, trainer: BaseTrainer,
                 dev_reader: BaseReader, configs: Config,
                 preprocessors: Optional[List[BaseProcessor]] = None,
                 evaluator: Optional[Evaluator] = None,
                 predictor: Optional[BaseProcessor] = None):
        self.resource = Resources()
        self.configs = configs

        train_reader.initialize(self.resource, self.configs.reader)

        if preprocessors is not None:
            for p in preprocessors:
                p.initialize(resources=self.resource,
                             configs=configs.preprocessor)
            self.preprocessors = preprocessors
        else:
            self.preprocessors = []

        self.train_reader = train_reader
        self.dev_reader = dev_reader
        self.trainer = trainer

        if predictor is not None:
            self.predictor = predictor

        if evaluator is not None:
            self.evaluator = evaluator
            self.evaluator.initialize(self.resource, self.configs.evaluator)

    def run(self):
        logging.info("Preparing the pipeline")
        self.prepare()
        logging.info("Initializing the trainer")

        # initialize the pipeline after prepare step, since prepare will update
        # the resources
        self.trainer.initialize(self.resource, self.configs)
        #TODO add assign manager here for trainer,predictor and evaluator, test if will cause bugs
        self.trainer.assign_manager(self.preprocessors[0]._process_manager)
        if self.predictor is not None:
            logger.info("Initializing the predictor")
            self.predictor.initialize(resources=self.resource, configs=self.configs)
            self.predictor.assign_manager(self.preprocessors[0]._process_manager)
        try:
            if self.evaluator is not None:
                self.evaluator.assign_manager(self.preprocessors[0]._process_manager)
        except:
            pass
        logging.info("The pipeline is training")
        self.train()
        self.finish()

    def prepare(self):
        prepare_pl = Pipeline()
        prepare_pl.set_reader(self.train_reader)
        for p in self.preprocessors:
            # prepare_pl.add(p)
            # TODO: ner debug, preprocessor needs config
            prepare_pl.add(p, config=self.configs["preprocessor"])
            # prepare_pl.add(p)
        prepare_pl.run(self.configs.config_data.train_path)
        # TODO: ner debug, transfer back the resource from prepare_pl to train_pl
        self.resource.update(**prepare_pl.resource.resources)

    def train(self):
        epoch = 0
        while True:
            epoch += 1
            for pack in self.train_reader.iter(
                    self.configs.config_data.train_path):
                for instance in pack.get_data(**self.trainer.data_request()):
                    self.trainer.consume(instance)

            self.trainer.epoch_finish_action(epoch)

            if self.trainer.validation_requested():
                dev_res = self._validate(epoch)
                self.trainer.validation_done()
                self.trainer.post_validation_action(dev_res)
            if self.trainer.stop_train():
                return

            logging.info("End of epoch %d", epoch)

    def _validate(self, epoch: int):
        validation_result = {"epoch": epoch}
        batch_size = 16
        current_batch_size = 0
        if self.predictor is not None:
            for pack in self.dev_reader.iter(
                    self.configs.config_data.val_path):
                predicted_pack = pack.view()
                current_batch_size += 1
                if current_batch_size == batch_size:
                    self.predictor.process(predicted_pack)
                    #TODO passed a epoch in maybe need to change back
                    self.evaluator.consume_next(predicted_pack, pack,epoch)
                    current_batch_size = 0
            validation_result["eval"] = self.evaluator.get_result()

        current_batch_test_size = 0
        if self.evaluator is not None:
            for pack in self.dev_reader.iter(
                    self.configs.config_data.test_path):
                predicted_pack = pack.view()
                current_batch_test_size += 1
                if current_batch_test_size == batch_size:
                    self.predictor.process(predicted_pack)
                # TODO passed a epoch in maybe need to change back
                    self.evaluator.consume_next(predicted_pack, pack,epoch)
                    current_batch_test_size = 0
            validation_result["test"] = self.evaluator.get_result()

        return validation_result

    def finish(self):
        self.train_reader.finish(self.resource)
        self.dev_reader.finish(self.resource)
        self.trainer.finish(self.resource)
        self.predictor.finish(self.resource)
