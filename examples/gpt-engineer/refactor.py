import instructor

from openai import OpenAI
from pydantic import Field, parse_file_as
from instructor import ResponseSchema
from generate import Program

client = instructor.from_openai(OpenAI())


class Diff(ResponseSchema):
    """
    Changes that must be correctly made in a program's code repository defined as a
    complete diff (Unified Format) file which will be used to `patch` the repository.

    Example:
      --- /path/to/original	timestamp
      +++ /path/to/new	timestamp
      @@ -1,3 +1,9 @@
      +This is an important
      +notice! It should
      +therefore be located at
      +the beginning of this
      +document!
      +
       This part of the
       document has stayed the
       same from version to
      @@ -8,13 +14,8 @@
       compress the size of the
       changes.
      -This paragraph contains
      -text that is outdated.
      -It will be deleted in the
      -near future.
      -
       It is important to spell
      -check this dokument. On
      +check this document. On
       the other hand, a
       misspelled word isn't
       the end of the world.
      @@ -22,3 +23,7 @@
       this paragraph needs to
       be changed. Things can
       be added after it.
      +
      +This paragraph contains
      +important new additions
      +to this document.
    """

    diff: str = Field(
        ...,
        description=(
            "Changes in a code repository correctly represented in 'diff' format, "
            "correctly escaped so it could be used in a JSON"
        ),
    )


def refactor(new_requirements: str, program: Program) -> Diff:
    program_description = "\n".join(
        [f"{code.file_name}\n[[[\n{code.body}\n]]]\n" for code in program.files]
    )
    completion = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        functions=[Diff.openai_schema],
        function_call={"name": Diff.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of refactor "
                "existing python repositories. You will name files correct, include "
                "__init__.py files and write correct python code, with correct imports. "
                "You'll deliver your changes in valid 'diff' format so that they could "
                "be applied using the 'patch' command. "
                "Make sure you put the correct line numbers, "
                "and that all lines that must be changed are correctly marked.",
            },
            {
                "role": "user",
                "content": new_requirements,
            },
            {
                "role": "user",
                "content": program_description,
            },
        ],
        max_tokens=1000,
    )
    return Diff.from_response(completion)


if __name__ == "__main__":
    program = parse_file_as(path="program.json", type_=Program)

    changes = refactor(
        new_requirements="Refactor this code to use flask instead.",
        program=program,
    )
    print(changes.diff)
    """
    --- readme.md
    +++ readme.md
    @@ -1,9 +1,9 @@
     # FastAPI App

    -This is a FastAPI app that provides some basic math functions.
    +This is a Flask app that provides some basic math functions.

     ## Usage

     To use this app, follow the instructions below:

     1. Install the required dependencies by running `pip install -r requirements.txt`.
    -2. Start the app by running `uvicorn main:app --reload`.
    +2. Start the app by running `flask run`.
     3. Open your browser and navigate to `http://localhost:5000/docs` to access the Swagger UI documentation.

     ## Example

     To perform a basic math operation, you can use the following curl command:

     ```bash
    -curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:8000/calculate
    +curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:5000/calculate
     ```

    --- main.py
    +++ main.py
    @@ -1,29 +1,29 @@
    -from fastapi import FastAPI
    -from pydantic import BaseModel
    +from flask import Flask, request, jsonify

    -app = FastAPI()
    +app = Flask(__name__)


    -class Operation(BaseModel):
    -    operation: str
    -    operands: list
    +@app.route('/calculate', methods=['POST'])
    +def calculate():
    +    data = request.get_json()
    +    operation = data.get('operation')
    +    operands = data.get('operands')


    -@app.post('/calculate')
    -async def calculate(operation: Operation):
    -    if operation.operation == 'add':
    -        result = sum(operation.operands)
    -    elif operation.operation == 'subtract':
    -        result = operation.operands[0] - sum(operation.operands[1:])
    -    elif operation.operation == 'multiply':
    +    if operation == 'add':
    +        result = sum(operands)
    +    elif operation == 'subtract':
    +        result = operands[0] - sum(operands[1:])
    +    elif operation == 'multiply':
             result = 1
    -        for operand in operation.operands:
    +        for operand in operands:
                 result *= operand
    -    elif operation.operation == 'divide':
    -        result = operation.operands[0]
    -        for operand in operation.operands[1:]:
    +    elif operation == 'divide':
    +        result = operands[0]
    +        for operand in operands[1:]:
                 result /= operand
         else:
             result = None
    -    return {'result': result}
    +    return jsonify({'result': result})

    --- requirements.txt
    +++ requirements.txt
    @@ -1,3 +1,2 @@
    -fastapi
    -uvicorn
    -pydantic
    +flask
    +flask-cors
    """

    with open("changes.diff", "w") as f:
        f.write(changes.diff)
