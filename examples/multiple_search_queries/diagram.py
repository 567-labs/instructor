import erdantic as erd

from segment_search_queries import MultiSearch

diagram = erd.create(MultiSearch)
diagram.draw("examples/segment_search_queries/schema.png")
