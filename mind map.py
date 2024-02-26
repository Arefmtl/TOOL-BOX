from graphviz import Digraph

def create_mind_map():
    # Create a new Digraph object
    dot = Digraph()

    # Add nodes and edges
    dot.node('A', 'Machine Learning Project')
    dot.node('B', 'Project Overview')
    dot.node('C', 'Experimental Results')
    dot.node('D', 'DBSCAN Results')
    dot.node('E', 'Louvain Method Results')
    dot.node('F', 'Observations')
    dot.node('G', 'Conclusion')
    dot.node('H', 'Future Directions')

    # Add edges between nodes
    dot.edges(['AB', 'AC', 'BC', 'CD', 'CE', 'CF', 'CG', 'CH'])

    return dot

# Create the mind map
dot = create_mind_map()

# Save the mind map as a PNG image
dot.render('machine_learning_project_mind_map', format='png', cleanup=True)
