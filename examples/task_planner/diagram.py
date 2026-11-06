import erdantic as erd

from task_planner_topological_sort import TaskPlan

diagram = erd.create(TaskPlan)
diagram.draw("examples/task_planner_topological_sort/schema.png")
