import erdantic as erd

from safe_sql import SQL

diagram = erd.create(SQL)
diagram.draw("examples/safe_sql/schema.png")
