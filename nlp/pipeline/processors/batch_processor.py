from abc import abstractmethod
from typing import Dict, Optional, Type, Union, List

from nlp.pipeline.data import slice_batch
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.batchers import ProcessingBatcher
from nlp.pipeline.data.ontology import Entry

__all__ = [
    "BatchProcessor",
]


class BatchProcessor(BaseProcessor):
    """
    The base class of processors that process data in batch.
    """
    def __init__(self):
        super().__init__()

        self.context_type = None
        self.input_info: Dict[Type[Entry], Union[List, Dict]] = {}

        self.batch_size = None
        self.batcher = None

    def initialize_batcher(self, hard_batch: bool = True):
        self.batcher = ProcessingBatcher(self.batch_size, hard_batch)

    def process(self, input_pack: DataPack, tail_instances: bool = False):
        if input_pack.meta.cache_state == self.component_name:
            input_pack = None  # type: ignore
        else:
            input_pack.meta.cache_state = self.component_name

        for batch in self.batcher.get_batch(input_pack,
                                            self.context_type,
                                            self.input_info,
                                            tail_instances=tail_instances):
            pred = self.predict(batch)
            self.pack_all(pred)
            self.finish_up_packs(-1)
        if len(self.batcher.current_batch_sources) == 0:
            self.finish_up_packs()

    @abstractmethod
    def predict(self, data_batch: Dict):
        """
        Make predictions for the input data_batch.

        Args:
              data_batch (Dict): A batch of instances in our dict format.

        Returns:
              The prediction results in dict format.
        """
        pass

    def pack_all(self, output_dict: Dict):
        start = 0
        for i in range(len(self.batcher.data_pack_pool)):
            output_dict_i = slice_batch(output_dict, start,
                                        self.batcher.current_batch_sources[i])
            self.pack(self.batcher.data_pack_pool[i], output_dict_i)
            start += self.batcher.current_batch_sources[i]

    @abstractmethod
    def pack(self, data_pack: DataPack, inputs) -> None:
        """
        Add corresponding fields to data_pack. Custom function of how
        to add the value back.

        Args:
            data_pack (DataPack): The data pack to add entries or fields to.
            inputs: The prediction results returned by :meth:`predict`. You
                need to add entries or fields corresponding to this prediction
                results to the ``data_pack``.
        """
        pass

    def finish_up_packs(self, end: Optional[int] = None):
        """
        Do finishing work for data packs in :attr:`data_pack_pool` from the
        beginning to ``end`` (``end`` is not included).

        Args:
            end (int): Will do finishing work for data packs in
                :attr:`data_pack_pool` from the beginning to ``end``
                (``end`` is not included). If `None`, will finish up all the
                packs in :attr:`data_pack_pool`.
        """
        if end is None:
            end = len(self.batcher.data_pack_pool)
        for pack in self.batcher.data_pack_pool[:end]:
            self.finish(pack)
        self.batcher.data_pack_pool = self.batcher.data_pack_pool[end:]
        self.batcher.current_batch_sources = \
            self.batcher.current_batch_sources[end:]
