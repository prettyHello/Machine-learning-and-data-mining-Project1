def DT_to_PNG(model, feature_names, file_name):
    """ Exports a DT to a PNG image file for inspection.
    
    Parameters
    ----------
        - model: a decision tree (class sklearn.tree.DecisionTreeClassifier)
        - feature_names: a list of feature names
        - file_name: name of file to be produced (without '.png' extension)
    
    Notes
    -----
    This function requires the pydot Python package and the Graphviz library.
    
    For more information about tree export, see http://scikit-learn.org/stable/
    modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
    
    """

    import pydot
    import string
    from sklearn import tree
    from sklearn.externals.six import StringIO
    
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_png('%s.png' % file_name)
