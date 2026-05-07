from dataclasses import dataclass
from typing import TypeAlias

from util.weights import Weights

ClientID: TypeAlias = str
ModelRev: TypeAlias = int
Timestamp: TypeAlias = float
WeightDiff: TypeAlias = Weights


class InvalidStateException(Exception):
	pass


@dataclass
class WeightsDelta:
	rev_a: ModelRev
	rev_b: ModelRev
	diff: WeightDiff


@dataclass
class Round:
	round_id: int
	rev_a: ModelRev
	rev_b: ModelRev
	deadline: Timestamp


@dataclass
class Message:
	pass


@dataclass
class FederationRequest(Message):
	client_id: ClientID
	local_model_rev: ModelRev


@dataclass
class FederationResponse(Message):
	global_model_rev: ModelRev


@dataclass
class UpdateRequest(Message):
	rev_a: ModelRev
	rev_b: ModelRev


@dataclass
class ModelPush(Message):
	model_rev: ModelRev
	weights: Weights


@dataclass
class DeltaPush(Message):
	delta: WeightsDelta


@dataclass
class RoundAnnounce(Message):
	round: Round


@dataclass
class RoundEnd(Message):
	round: Round
	success: bool
	delta: WeightsDelta | None
