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
from typing import Any, Dict

from forte.pipeline import Pipeline
from forte.data.readers import bc5cdr_reader
from examples.bc5cdr.bc5cdr_trainer import BC5CDRTrainer
from examples.bc5cdr.biobert_ner_predictor import BC5CDRPredictor
from examples.bc5cdr.bc5cdr_evaluator import BC5CDREvaluator
from examples.bc5cdr.DummyPreprocessor import BC5CDRDummyProcessor
from forte.train_pipeline import TrainPipeline
import yaml
from examples.bc5cdr.biobert_ner_predictor import *

def pack_example(input_path, output_path):
    """
    This example read data from input path and serialize to output path.
    Args:
        input_path:
        output_path:

    Returns:

    """

    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    config_preprocess = yaml.safe_load(open("config_preprocessor.yml", "r"))

    config = Config({}, default_hparams=None)
    config.add_hparam('config_data', config_data)
    config.add_hparam('config_model', config_model)
    config.add_hparam('preprocessor', config_preprocess)
    config.add_hparam('reader', {})
    config.add_hparam('evaluator', {})

    print("Biobert BC5CDR training example.")

    reader = bc5cdr_reader.BC5CDRReader()


    dummprocessor = BC5CDRDummyProcessor()

    bc5cdr_trainer = BC5CDRTrainer()
    bc5cdr_predictor = BC5CDRPredictor()
    bc5cdr_evaluator = BC5CDREvaluator()
    train_pipe = TrainPipeline(train_reader=reader, trainer=bc5cdr_trainer,
                               dev_reader=reader,configs=config,
                               predictor=bc5cdr_predictor,preprocessors=[dummprocessor],
                               evaluator=bc5cdr_evaluator)
    train_pipe.run()




def main(data_path: str):
    pack_output = 'pack_out'
    # multipack_output = 'multi_out'

    pack_example(data_path, pack_output)

if __name__ == '__main__':
    main("data/train/")
