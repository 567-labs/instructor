from erdantic import erd

from query_planner_execution import QueryPlan

diagram = erd.create(QueryPlan)
diagram.draw("examples/query_planner_execution/schema.png")
