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

import logging

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.models.ner.utils import load_glove_embedding, normalize_digit_word
from forte.processors import Alphabet
from forte.processors import VocabularyProcessor
from ft.onto.base_ontology import Token, Sentence

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.models.ner.utils import load_glove_embedding, normalize_digit_word
from forte.processors import VocabularyProcessor

class BC5CDRDummyProcessor(VocabularyProcessor):
    """
    Vocabulary Processor for the datasets of CoNLL data
    Create the vocabulary for the word, character, pos tag, chunk id and ner
    tag
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = True
    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        pass

    def _process(self, data_pack: DataPack):
        """
        Process the data pack to collect vocabulary information.

        Args:
            data_pack: The ner data to create vocabulary with.

        Returns:

        """
        # for data_pack in input_pack:
        for instance in data_pack.get_data(
                context_type=Sentence,
                request={Token: ["ner"]}):
            continue

    def finish(self, resource: Resources):
        # if a singleton is in pre-trained embedding dict,
        # set the count to min_occur + c
        pass