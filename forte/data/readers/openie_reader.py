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
"""
The reader that reads Open-IE extractions data into data pack.
Format:
https://github.com/gabrielStanovsky/oie-benchmark/tree/master/oie_corpus
"""
import logging
import os
from typing import Iterator, List

from forte.common.exception import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Sentence, PredicateMention, \
    PredicateArgument, PredicateLink, Document, Token

__all__ = [
    "OpenIEReader"
]


class OpenIEReader(PackReader):
    r""":class:`OpenIEReader` is designed to read in the Open IE dataset used
        by Open Information Extraction task. The related paper can be found
        `here
        <https://gabrielstanovsky.github.io/assets/papers/emnlp16a/paper.pdf>`__.
        The related source code for generating this dataset can be found
        `here
        <https://github.com/gabrielStanovsky/oie-benchmark>`__.
        To use this Reader, you must follow the dataset format. Each line in
        the dataset should contain following fields:

        .. code-block:: none

            <sentence>\t<predicate_head>\t<full_predicate>\t<arg1>\t<arg2>....

        You can also find the dataset format `here
        <https://github.com/gabrielStanovsky/oie-benchmark/tree/master/oie_corpus>`__.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if configs.oie_file_extension is None:
            raise ProcessorConfigError(
                "Configuration oie_file_extension not provided.")

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""Should be called with param ``oie_directory`` which is a path to a
        folder containing json files.

        Args:
            args: args[0] is the directory to the open ie files.
            kwargs:

        Returns: Iterator over files in the path with oie extensions.
        """
        oie_directory: str = args[0]
        oie_file_extension: str = self.configs.oie_file_extension
        logging.info("Reading dataset from %s with extension %s",
                     oie_directory, oie_file_extension)
        return dataset_path_iterator(oie_directory, oie_file_extension)

    def _cache_key_function(self, oie_file: str) -> str:
        return os.path.basename(oie_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack: DataPack = self.new_pack()
        text: str = ""
        offset: int = 0

        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    oie_component: List[str] = line.split("\t")
                    sentence: str = oie_component[0]

                    # Add sentence.
                    Sentence(pack, offset, offset + len(sentence))
                    offset += len(sentence) + 1
                    text += sentence + " "

                    head_predicate: str = oie_component[1]
                    full_predicate: str = oie_component[2]

                    # Add head predicate.
                    token: Token = Token(pack,
                                         offset,
                                         offset + len(head_predicate))
                    offset += len(head_predicate) + 1
                    text += head_predicate + " "

                    # Add full predicate.
                    predicate_mention: PredicateMention = PredicateMention(pack,
                                                         offset,
                                                         offset
                                                         + len(full_predicate))
                    predicate_mention.headword = token
                    offset += len(full_predicate) + 1
                    text += full_predicate + " "

                    for arg in oie_component[3:]:
                        # Add predicate argument.
                        predicate_arg: PredicateArgument = \
                            PredicateArgument(pack,
                                              offset,
                                              offset + len(arg))
                        offset += len(arg) + 1
                        text += arg + " "

                        # Add predicate link.
                        PredicateLink(pack, predicate_mention, predicate_arg)

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        pack.pack_name = file_path

        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()

        # Add OIE dataset file extension. The default is '.oie'
        config.update({
            'oie_file_extension': 'oie'
        })
        return config
