from graphviz import Digraph
     
def draw_graph(node):
    '''Draws a node's dependency graph with graphviz. 
       Note: This looks like a depth-first, pre-order traversal, but it's a DAG rather than a tree. 
             e.g. one node can be used in multiple downstream nodes.'''
    def _draw_node(node):
        '''Draws / adds a single node to the graph.'''
        # Don't add duplicate nodes to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add it once
        if f'\t{id(node)}' in dot.body: return
        
        # Add the node with the appropriate text
        if node.parent_op is None:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.value:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">input</FONT></TD></TR>
            </TABLE>>'''
        else:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.value:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD BGCOLOR="#c2ebff"><FONT COLOR="#004261" FACE="Courier" POINT-SIZE="12">{node.parent_op}</FONT></TD></TR>
            </TABLE>>'''
        dot.node(str(id(node)), node_text)
            
    def _draw_edge(parent, node):
        '''Draws / adds a single directed edge to the graph (parent -> node).'''
        # Don't add duplicate edges to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add edges to its parents once
        if f'\t{id(parent)} -> {id(node)}' in dot.body: return
        
        # Add the edge
        dot.edge(str(id(parent)), str(id(node)))
    
    def _draw_parents(node):
        '''Traverses recursively, drawing the parent at the child's step (in order to draw the edge).'''
        for parent in node.parents:
            _draw_node(parent)
            _draw_edge(parent, node)
            _draw_parents(parent)
   
    dot = Digraph(graph_attr={'rankdir': 'BT'}, node_attr={'shape': 'plaintext'})
    _draw_node(node)                             # Draw the root / output      
    _draw_parents(node)                          # Draw the rest of the graph
    
    return dot