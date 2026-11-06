import pytest
import instructor
from openai import OpenAI
import pprint


@pytest.fixture
def client():
    return instructor.from_openai(OpenAI())


def log_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


def log_kwargs_1(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


def log_kwargs_2(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


hook_names = [item.value for item in instructor.hooks.HookName]
hook_enums = [instructor.hooks.HookName(hook_name) for hook_name in hook_names]
hook_functions = [log_kwargs, log_kwargs_1, log_kwargs_2]
hook_object = instructor.hooks.Hooks()


@pytest.mark.parametrize("hook_name", hook_names)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_on_method_str(
    client: instructor.Instructor, hook_name: str, num_functions: int
):
    functions_to_add = hook_functions[:num_functions]
    hook_enum = hook_object.get_hook_name(hook_name)

    assert hook_enum not in client.hooks._handlers

    for func in functions_to_add:
        client.on(hook_name, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    for func in functions_to_add:
        assert func in client.hooks._handlers[hook_enum]


@pytest.mark.parametrize("hook_enum", hook_enums)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_on_method_enum(
    client: instructor.Instructor,
    hook_enum: instructor.hooks.HookName,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]
    assert hook_enum not in client.hooks._handlers

    for func in functions_to_add:
        client.on(hook_enum, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    for func in functions_to_add:
        assert func in client.hooks._handlers[hook_enum]


@pytest.mark.parametrize("hook_name", hook_names)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_off_method_str(
    client: instructor.Instructor,
    hook_name: str,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]
    hook_enum = hook_object.get_hook_name(hook_name)
    assert hook_enum not in client.hooks._handlers

    for func in functions_to_add:
        client.on(hook_name, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    for func in functions_to_add:
        client.off(hook_name, func)
        if client.hooks._handlers.get(hook_enum):
            assert func not in client.hooks._handlers[hook_enum]
        else:
            assert hook_enum not in client.hooks._handlers

    assert hook_enum not in client.hooks._handlers


@pytest.mark.parametrize("hook_enum", hook_enums)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_off_method_enum(
    client: instructor.Instructor,
    hook_enum: instructor.hooks.HookName,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]
    assert hook_enum not in client.hooks._handlers
    for func in functions_to_add:
        client.on(hook_enum, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    for func in functions_to_add:
        client.off(hook_enum, func)
        if client.hooks._handlers.get(hook_enum):
            assert func not in client.hooks._handlers[hook_enum]
        else:
            assert hook_enum not in client.hooks._handlers

    assert hook_enum not in client.hooks._handlers


@pytest.mark.parametrize("hook_name", hook_names)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_clear_method_str(
    client: instructor.Instructor,
    hook_name: str,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]

    for func in functions_to_add:
        client.on(hook_name, func)

    hook_enum = hook_object.get_hook_name(hook_name)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    client.clear(hook_name)
    assert hook_enum not in client.hooks._handlers


@pytest.mark.parametrize("hook_enum", hook_enums)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_clear_method(
    client: instructor.Instructor,
    hook_enum: instructor.hooks.HookName,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]

    for func in functions_to_add:
        client.on(hook_enum, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    client.clear(hook_enum)
    assert hook_enum not in client.hooks._handlers


@pytest.mark.parametrize("hook_enum", hook_enums)
@pytest.mark.parametrize("num_functions", [1, 2, 3])
def test_clear_no_args(
    client: instructor.Instructor,
    hook_enum: instructor.hooks.HookName,
    num_functions: int,
):
    functions_to_add = hook_functions[:num_functions]

    for func in functions_to_add:
        client.on(hook_enum, func)

    assert hook_enum in client.hooks._handlers
    assert len(client.hooks._handlers[hook_enum]) == num_functions

    client.clear()
    assert hook_enum not in client.hooks._handlers
