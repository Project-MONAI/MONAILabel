from typing import Dict, Hashable, List, Optional

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.enums import TransformBackends

from monailabel.interactions import ProtoSegmentation, Mapping, InterestDefinition, Seeding

__all__ = ["InteractionProtocolD", "InteractionProtocolDict", "InteractionProtocold"]


# TODO: Discuss with @ericspod and @rijobro in regards to the design of this class and MetaTensor
class InteractionProtocold(MapTransform):
    backend: List[TransformBackends] = []

    def __init__(
        self,
        keys: KeysCollection,
        proto_segmentation: ProtoSegmentation,
        interest_definition: InterestDefinition,
        seeding: Seeding,
        mapping: Mapping,
        proto_segmentation_keys: Optional[KeysCollection] = None,
        proto_segmentation_postfix: str = "proto_seg",
        interest_definition_keys: Optional[KeysCollection] = None,
        interest_definition_postfix: str = "interest_map",
        seeding_keys: Optional[KeysCollection] = None,
        seeding_postfix: str = "seeds",
        mapping_keys: Optional[KeysCollection] = None,
        mapping_postfix: str = "interaction_map",
        overwriting: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.proto_segmentation = proto_segmentation
        self.proto_segmentation_postfix = ensure_tuple_rep(proto_segmentation_postfix, len(self.keys))
        self.proto_segmentation_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if proto_segmentation_keys is None
            else ensure_tuple(proto_segmentation_keys)
        )
        if len(self.keys) != len(self.proto_segmentation_keys):
            raise ValueError("proto_segmentation_keys should have the same length as keys.")

        self.interest_definition = interest_definition
        self.interest_definition_postfix = ensure_tuple_rep(interest_definition_postfix, len(self.keys))
        self.interest_definition_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if interest_definition_keys is None
            else ensure_tuple(interest_definition_keys)
        )
        if len(self.keys) != len(self.interest_definition_keys):
            raise ValueError("interest_definition_keys should have the same length as keys.")

        self.seeding = seeding
        self.seeding_postfix = ensure_tuple_rep(seeding_postfix, len(self.keys))
        self.seeding_keys = (
            ensure_tuple_rep(None, len(self.keys)) if seeding_keys is None else ensure_tuple(seeding_keys)
        )
        if len(self.keys) != len(self.seeding_keys):
            raise ValueError("seeding_keys should have the same length as keys.")

        self.mapping = mapping
        self.mapping_postfix = ensure_tuple_rep(mapping_postfix, len(self.keys))
        self.mapping_keys = (
            ensure_tuple_rep(None, len(self.keys)) if mapping_keys is None else ensure_tuple(mapping_keys)
        )
        if len(self.keys) != len(self.mapping_keys):
            raise ValueError("mapping_keys should have the same length as keys.")

        self._define_backend()

        self.overwriting = overwriting

    def _define_backend(self) -> None:
        # Define the available backends for this Transform based on the backend of the sub-transforms
        self.backend = list(
            set(self.proto_segmentation.backend)
            & set(self.interest_definition.backend)
            & set(self.seeding.backend)
            & set(self.mapping.backend)
        )

    def _update_data(
        self,
        data: Mapping[Hashable, NdarrayOrTensor],
        key: Hashable,
        meta_data: NdarrayOrTensor,
        meta_key: str,
        meta_key_postfix: str,
    ) -> None:
        meta_key = meta_key or f"{key}_{meta_key_postfix}"

        if meta_key in data and not self.overwriting:
            raise KeyError(f"Meta data with key {meta_key} already exists and overwriting=False.")

        data[meta_key] = meta_data

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for (
            key,
            proto_segmentation_postfix,
            proto_segmentation_key,
            interest_definition_postfix,
            interest_definition_keys,
            seeding_postfix,
            seeding_keys,
            mapping_postfix,
            mapping_keys,
        ) in self.key_iterator(
            d,
            self.proto_segmentation_postfix,
            self.proto_segmentation_keys,
            self.interest_definition_postfix,
            self.interest_definition_keys,
            self.seeding_postfix,
            self.seeding_keys,
            self.mapping_postfix,
            self.mapping_keys,
        ):
            proto_segmentation = self.proto_segmentation(d[key])
            interest_definition = self.interest_definition(d[key], proto_segmentation=proto_segmentation)
            seeding = self.seeding(d[key], interest_definition=interest_definition)
            mapping = self.mapping(d[key], seeding=seeding)

            self._update_data(
                data=d,
                key=key,
                meta_data=proto_segmentation,
                meta_key=proto_segmentation_key,
                meta_key_postfix=proto_segmentation_postfix,
            )

            self._update_data(
                data=d,
                key=key,
                meta_data=interest_definition,
                meta_key=interest_definition_keys,
                meta_key_postfix=interest_definition_postfix,
            )

            self._update_data(
                data=d, key=key, meta_data=seeding, meta_key=seeding_keys, meta_key_postfix=seeding_postfix
            )

            self._update_data(
                data=d, key=key, meta_data=mapping, meta_key=mapping_keys, meta_key_postfix=mapping_postfix
            )

        return d


InteractionProtocolD = InteractionProtocolDict = InteractionProtocold
