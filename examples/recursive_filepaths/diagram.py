import erdantic as erd

from parse_recursive_paths import DirectoryTree

diagram = erd.create(DirectoryTree)
diagram.draw("examples/parse_recursive_paths/schema.png")
