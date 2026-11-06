from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Union
import numpy as np
import json
from pydantic import ValidationError
from pprint import pprint
import models as m


class Status(Enum):
    IS_JSON = "_is_json_"
    IS_VALID = "_is_valid_"
    VALIDATION_ERROR = "_validation_error_"


class StreamingAccumulatorManager:
    def __init__(self):
        self.accumulator = defaultdict(StreamingAccumulator)

    def validate_string(self, json_string: str, index: int) -> None:
        try:
            obj = json.loads(json_string)
            self.accumulator[Status.IS_JSON.value].update(index, True)
            try:
                # Replace this line with your validation logic
                obj = m.MultiSearch.model_validate(obj)
                self.update(index, obj.model_dump())
                self.accumulator[Status.IS_VALID.value].update(index, True)
            except ValidationError as e:
                self.accumulator[Status.IS_VALID.value].update(index, False)
                self.process_validation_error(e, index)
        except json.JSONDecodeError:
            self.accumulator[Status.IS_JSON.value].update(index, False)

    def process_validation_error(self, error, index):
        for err in error.errors():
            path = (
                "$."
                + ".".join(
                    [str(x) if not isinstance(x, int) else "[*]" for x in err["loc"]]
                )
                + "."
                + err["type"]
            )
            self.accumulator[Status.VALIDATION_ERROR.value].update(index, path)

    def update(self, index, data: Any, path: str = "$") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}"
                self.update(index, value, new_path)
        elif isinstance(data, list):
            new_path = f"{path}[*]"
            for value in data:
                self.update(index, value, new_path)
            length_path = f"{path}.length"
            self.accumulator[length_path].update(index, len(data))
        elif isinstance(data, Enum):
            enum_path = f"{path}.enum"
            self.accumulator[enum_path].update(index, data.value)
        else:
            self.accumulator[path].update(index, data)

    def summarize(self) -> dict[str, dict]:
        return {k: v.summarize(key_name=k) for k, v in self.accumulator.items()}


class StreamingAccumulator:
    def __init__(self):
        self.counter = Counter()
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0
        self.squared_sum = 0
        self.unique_values = set()
        self.missing_values = 0
        self.str_min_length = float("inf")
        self.str_max_length = float("-inf")
        self.str_sum_length = 0
        self.str_squared_sum_length = 0
        self.value = []
        self.str_length = []
        self.reverse_lookup = defaultdict(list)

    def update(self, index: Any, value: Any) -> None:
        if isinstance(value, (int, str, bool)):
            self.counter[value] += 1
            self.unique_values.add(value)
            self.value.append(value)
            self.reverse_lookup[value].append(index)
        if value is None or value == "":
            self.missing_values += 1
            return
        if isinstance(value, (int, float)):
            self.min = min(self.min, value)
            self.max = max(self.max, value)
            self.sum += value
            self.squared_sum += value**2
        if isinstance(value, str):
            str_len = len(value)
            self.str_length.append(str_len)
            self.str_min_length = min(self.str_min_length, str_len)
            self.str_max_length = max(self.str_max_length, str_len)
            self.str_sum_length += str_len
            self.str_squared_sum_length += str_len**2

    def summarize(self, key_name=None) -> dict[str, Union[int, float, dict]]:
        if key_name is None:
            key_name = ""
        n = sum(self.counter.values())
        summaries = {}
        summaries["counter"] = self.counter
        summaries["unique_count"] = len(self.unique_values)
        summaries["missing_values"] = self.missing_values
        summaries["_reverse_lookup"] = dict(self.reverse_lookup)
        if n > 0:
            if all(isinstance(value, (bool)) for value in self.unique_values):
                summaries["mean"] = self.sum / n
                return summaries
            if all(isinstance(value, (int, float)) for value in self.unique_values):
                summaries["min"] = self.min
                summaries["max"] = self.max
                summaries["mean"] = self.sum / n
                summaries["std"] = np.sqrt(self.squared_sum / n - (self.sum / n) ** 2)
                return summaries
            if all(isinstance(value, str) for value in self.unique_values):
                summaries["str_min_length"] = self.str_min_length
                summaries["str_max_length"] = self.str_max_length
                summaries["str_mean_length"] = self.str_sum_length / n
                summaries["str_std_length"] = np.sqrt(
                    self.str_squared_sum_length / n - (self.str_sum_length / n) ** 2
                )
                return summaries
        return summaries


if __name__ == "__main__":
    eval_manager = StreamingAccumulatorManager()

    with open("test.jsonl") as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            eval_manager.validate_string(line, ii)

    pprint(eval_manager.summarize())
