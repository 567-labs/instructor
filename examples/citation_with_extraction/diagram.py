import erdantic as erd

from citation_fuzzy_match import QuestionAnswer

diagram = erd.create(QuestionAnswer)
diagram.draw("examples/citation_fuzzy_match/schema.png")
