import pytest
from typing import Iterable, Dict, Set, Tuple
from collections.abc import Iterable as AbcIterable
from instructor.v2.core.response_model import prepare_response_model
from pydantic import BaseModel

def test_dict_simple_type():
    Model = prepare_response_model(dict[str, int])
    assert issubclass(Model, BaseModel)
    schema = Model.model_json_schema()
    assert "content" in schema["properties"]
    assert schema["properties"]["content"]["type"] == "object"
    
    # Test unpacking
    obj = Model.model_validate_json('{"content": {"a": 1}}')
    assert obj.content == {"a": 1}

def test_set_simple_type():
    Model = prepare_response_model(set[int])
    assert issubclass(Model, BaseModel)
    schema = Model.model_json_schema()
    assert schema["properties"]["content"]["type"] == "array"
    assert schema["properties"]["content"].get("uniqueItems") is True

def test_iterable_str():
    Model = prepare_response_model(Iterable[str])
    assert hasattr(Model, "extract_cls_task_type")
    
    val = Model.extract_cls_task_type('{"content": "hello"}')
    assert val == "hello"
    assert isinstance(val, str)

def test_iterable_dict():
    Model = prepare_response_model(AbcIterable[dict[str, int]])
    
    val = Model.extract_cls_task_type('{"content": {"k": 1}}')
    assert val == {"k": 1}
    assert isinstance(val, dict)
