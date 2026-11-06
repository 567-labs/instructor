"""
Proof of Concept for a task planning and execution system using
OpenAIs Functions and topological sort, based on the idea in
query_planner_execution.py.py.

Additionally: There are also cases where the "pure" recursive approach has advantages;
If subtasks for different parent tasks that start in parallel have different runtimes,
we will wait unnecessarily with my current implementation.

Added by Jan Philipp Harries / @jpdus
"""

import asyncio
from collections.abc import Generator

from openai import OpenAI

from pydantic import Field, BaseModel

import instructor

client = instructor.from_openai(OpenAI())


class TaskResult(BaseModel):
    task_id: int
    result: str


class TaskResults(BaseModel):
    results: list[TaskResult]


class Task(BaseModel):
    """
    Class representing a single task in a task plan.
    """

    id: int = Field(..., description="Unique id of the task")
    task: str = Field(
        ...,
        description="""Contains the task in text form. If there are multiple tasks,
        this task can only be executed when all dependant subtasks have been answered.""",
    )
    subtasks: list[int] = Field(
        default_factory=list,
        description="""List of the IDs of subtasks that need to be answered before
        we can answer the main question. Use a subtask when anything may be unknown
        and we need to ask multiple questions to get the answer.
        Dependencies must only be other tasks.""",
    )

    async def aexecute(self, with_results: TaskResults) -> TaskResult:
        """
        Executes the task by asking the question and returning the answer.
        """

        # We do nothing with the subtask answers, since this is an example however
        # we could use intermediate results to compute the answer to the main task.
        return TaskResult(task_id=self.id, result=f"`{self.task}`")


class TaskPlan(BaseModel):
    """
    Container class representing a tree of tasks and subtasks.
    Make sure every task is in the tree, and every task is done only once.
    """

    task_graph: list[Task] = Field(
        ...,
        description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.",
    )

    def _get_execution_order(self) -> list[int]:
        """
        Returns the order in which the tasks should be executed using topological sort.
        Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
        """
        tmp_dep_graph = {item.id: set(item.subtasks) for item in self.task_graph}

        def topological_sort(
            dep_graph: dict[int, set[int]],
        ) -> Generator[set[int], None, None]:
            while True:
                ordered = set(item for item, dep in dep_graph.items() if len(dep) == 0)
                if not ordered:
                    break
                yield ordered
                dep_graph = {
                    item: (dep - ordered)
                    for item, dep in dep_graph.items()
                    if item not in ordered
                }
            if len(dep_graph) != 0:
                raise ValueError(
                    f"Circular dependencies exist among these items: {{{', '.join(f'{key}:{value}' for key, value in dep_graph.items())}}}"
                )

        result = []
        for d in topological_sort(tmp_dep_graph):
            result.extend(sorted(d))
        return result

    async def execute(self) -> dict[int, TaskResult]:
        """
        Executes the tasks in the task plan in the correct order using asyncio and chunks with answered dependencies.
        """
        execution_order = self._get_execution_order()
        tasks = {q.id: q for q in self.task_graph}
        task_results = {}
        while True:
            ready_to_execute = [
                tasks[task_id]
                for task_id in execution_order
                if task_id not in task_results
                and all(
                    subtask_id in task_results for subtask_id in tasks[task_id].subtasks
                )
            ]
            # prints chunks to visualize execution order
            print(ready_to_execute)
            computed_answers = await asyncio.gather(
                *[
                    q.aexecute(
                        with_results=TaskResults(
                            results=[
                                result
                                for result in task_results.values()
                                if result.task_id in q.subtasks
                            ]
                        )
                    )
                    for q in ready_to_execute
                ]
            )
            for answer in computed_answers:
                task_results[answer.task_id] = answer
            if len(task_results) == len(execution_order):
                break
        return task_results


Task.model_rebuild()
TaskPlan.model_rebuild()


def task_planner(question: str) -> TaskPlan:
    messages = [
        {
            "role": "system",
            "content": "You are a world class task planning algorithm capable of breaking apart tasks into dependant subtasks, such that the answers can be used to enable the system completing the main task. Do not complete the user task, simply provide a correct compute graph with good specific tasks to ask and relevant subtasks. Before completing the list of tasks, think step by step to get a better understanding the problem.",
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4-0613",
        temperature=0,
        response_model=TaskPlan,
        messages=messages,
        max_tokens=1000,
    )
    root = TaskPlan.from_response(completion)

    return root


if __name__ == "__main__":
    plan = task_planner(
        "What is the difference in populations betweend the adjacent countries of Jan's home country and the adjacent countries of Jason's home country?"
    )
    print(plan.model_dump_json(indent=2))
    {
        "task_graph": [
            {"id": 1, "subtasks": [], "task": "Identify Jan's home country"},
            {
                "id": 2,
                "subtasks": [1],
                "task": "Identify the adjacent countries of Jan's home country",
            },
            {
                "id": 3,
                "subtasks": [2],
                "task": "Calculate the total population of the adjacent "
                "countries of Jan's home country",
            },
            {"id": 4, "subtasks": [], "task": "Identify Jason's home country"},
            {
                "id": 5,
                "subtasks": [4],
                "task": "Identify the adjacent countries of Jason's home country",
            },
            {
                "id": 6,
                "subtasks": [5],
                "task": "Calculate the total population of the adjacent "
                "countries of Jason's home country",
            },
            {
                "id": 7,
                "subtasks": [3, 6],
                "task": "Calculate the difference in populations between the "
                "adjacent countries of Jan's home country and the "
                "adjacent countries of Jason's home country",
            },
        ]
    }
