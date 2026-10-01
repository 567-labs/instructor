import instructor

from openai import OpenAI
from pydantic import Field
from instructor import ResponseSchema

client = instructor.from_openai(OpenAI())


class File(ResponseSchema):
    """
    Correctly named file with contents.
    """

    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    body: str = Field(..., description="Correct contents of a file")

    def save(self):
        with open(self.file_name, "w") as f:
            f.write(self.body)


class Program(ResponseSchema):
    """
    Set of files that represent a complete and correct program
    """

    files: list[File] = Field(..., description="List of files")


def develop(data: str) -> Program:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        functions=[Program.openai_schema],
        function_call={"name": Program.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of writing correct python scripts and modules. You will name files correct, include __init__.py files and write correct python code. with correct imports.",
            },
            {
                "role": "user",
                "content": data,
            },
        ],
        max_tokens=1000,
    )
    return Program.from_response(completion)


if __name__ == "__main__":
    program = develop(
        """
        Create a fastapi app with a readme.md file and a main.py file with
        some basic math functions. the datamodels should use pydantic and
        the main.py should use fastapi. the readme.md should have a title
        and a description. The readme should contain some helpful information
        and a curl example"""
    )

    for file in program.files:
        print(file.file_name)
        print("-")
        print(file.body)
        print("\n\n\n")
    """
    readme.md
    -
    # FastAPI App

    This is a FastAPI app that provides some basic math functions.

    ## Usage

    To use this app, follow the instructions below:

    1. Install the required dependencies by running `pip install -r requirements.txt`.
    2. Start the app by running `uvicorn main:app --reload`.
    3. Open your browser and navigate to `http://localhost:8000/docs` to access the Swagger UI documentation.

    ## Example

    To perform a basic math operation, you can use the following curl command:

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:8000/calculate
    ```





    main.py
    -
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()


    class Operation(BaseModel):
        operation: str
        operands: list


    @app.post('/calculate')
    async def calculate(operation: Operation):
        if operation.operation == 'add':
            result = sum(operation.operands)
        elif operation.operation == 'subtract':
            result = operation.operands[0] - sum(operation.operands[1:])
        elif operation.operation == 'multiply':
            result = 1
            for operand in operation.operands:
                result *= operand
        elif operation.operation == 'divide':
            result = operation.operands[0]
            for operand in operation.operands[1:]:
                result /= operand
        else:
            result = None
        return {'result': result}





    requirements.txt
    -
    fastapi
    uvicorn
    pydantic
    """

    with open("program.json", "w") as f:
        f.write(Program.parse_obj(program).json())
