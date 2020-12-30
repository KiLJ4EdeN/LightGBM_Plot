from lightgbm import LGBMModel, Booster
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as image


def _float2str(value, precision=None):
    return ("{0:.{1}f}".format(value, precision)
            if precision is not None and not isinstance(value, str)
            else str(value))


def _to_graphviz(tree_info, show_info, feature_names, precision=3,
                 orientation='horizontal', **kwargs):
    """Convert specified tree to graphviz instance.

    See:
      - https://graphviz.readthedocs.io/en/stable/api.html#digraph
    """

    def add(root, total_count, parent=None, decision=None, fcolor=None, first_node=False):
        """Recursively add node or edge."""
        if 'split_index' in root:  # non-leaf
            l_dec = 'yes'
            r_dec = 'no'
            if root['decision_type'] == '<=':
                lte_symbol = "&#8804;"
                operator = lte_symbol
            elif root['decision_type'] == '==':
                operator = "="
            else:
                raise ValueError('Invalid decision type in tree model.')
            name = 'split{0}'.format(root['split_index'])
            if feature_names is not None:
                label = '<B>{0}</B> {1} '.format(feature_names[root['split_feature']], operator)
            else:
                label = 'feature <B>{0}</B> {1} '.format(root['split_feature'], operator)
            label += '<B>{0}</B>'.format(_float2str(root['threshold'], precision))
            for info in ['split_gain', 'internal_value', 'internal_weight', "internal_count", "data_percentage"]:
                if info in show_info:
                    output = info.split('_')[-1]
                    if info in {'split_gain', 'internal_value', 'internal_weight'}:
                        label += '<br/>{0} {1}'.format(_float2str(root[info], precision), output)
                    elif info == 'internal_count':
                        label += '<br/>{0}: {1}'.format(output, root[info])
                    elif info == "data_percentage":
                        label += '<br/>{0}% of data'.format(_float2str(root['internal_count'] / total_count * 100, 2))

            fillcolor = "red"
            style = ""
            # if constraints:
            if fcolor is not None:
                # if constraints[root['split_feature']] == 1:
                if fcolor:
                    fillcolor = "#ddffdd"  # light red
                # if constraints[root['split_feature']] == -1:
                if not fcolor:
                    fillcolor = "#ffdddd"  # light green
                style = "filled"

            # have a condition for first nodes
            if first_node:
                fillcolor = "#dddddd"
                style = "filled"


            label = "<" + label + ">"
            graph.node(name, label=label, shape="rectangle", style=style, fillcolor=fillcolor)
            add(root['left_child'], total_count, name, l_dec, fcolor=True)
            add(root['right_child'], total_count, name, r_dec, fcolor=False)
        else:  # leaf
            name = 'leaf{0}'.format(root['leaf_index'])
            label = 'leaf {0}: '.format(root['leaf_index'])
            label += '<B>{0}</B>'.format(_float2str(root['leaf_value'], precision))
            if 'leaf_weight' in show_info:
                label += '<br/>{0} weight'.format(_float2str(root['leaf_weight'], precision))
            if 'leaf_count' in show_info:
                label += '<br/>count: {0}'.format(root['leaf_count'])
            if "data_percentage" in show_info:
                label += '<br/>{0}% of data'.format(_float2str(root['leaf_count'] / total_count * 100, 2))
            label = "<" + label + ">"
            graph.node(name, label=label, style="filled", fillcolor="#ddddff")
        if parent is not None:
            if decision == "yes":
              graph.edge(parent, name, decision, color="green")
            else:
              graph.edge(parent, name, decision, color="red")

    graph = Digraph(**kwargs)
    rankdir = "LR" if orientation == "horizontal" else "TB"
    graph.attr("graph", nodesep="0.05", ranksep="0.3", rankdir=rankdir)
    if "internal_count" in tree_info['tree_structure']:
        add(tree_info['tree_structure'], tree_info['tree_structure']["internal_count"], first_node=True)
    else:
        raise Exception("Cannot plot trees with no split")


    # Here is the legend thingy not important
    # if constraints:
    if True:
        # "#ddffdd" is light green, "#ffdddd" is light red
        legend = """<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
             <TR>
              <TD COLSPAN="2"><B>Monotone constraints</B></TD>
             </TR>
             <TR>
              <TD>Increasing</TD>
              <TD BGCOLOR="#ddffdd"></TD>
             </TR>
             <TR>
              <TD>Decreasing</TD>
              <TD BGCOLOR="#ffdddd"></TD>
             </TR>
             <TR>
              <TD>Leaf Node</TD>
              <TD BGCOLOR="#ddddff"></TD>
             </TR>
             <TR>
              <TD>Root Node</TD>
              <TD BGCOLOR="#dddddd"></TD>
             </TR>
            </TABLE>
           >"""
        graph.node("legend", label=legend, shape="rectangle", color="black")
    return graph


def create_tree_digraph(booster, tree_index=0, show_info=None, precision=3,
                        orientation='horizontal', **kwargs):
    """Create a digraph representation of specified tree.

    Each node in the graph represents a node in the tree.

    Non-leaf nodes have labels like ``Column_10 <= 875.9``, which means
    "this node splits on the feature named "Column_10", with threshold 875.9".

    Leaf nodes have labels like ``leaf 2: 0.422``, which means "this node is a
    leaf node, and the predicted value for records that fall into this node
    is 0.422". The number (``2``) is an internal unique identifier and doesn't
    have any special meaning.

    .. note::

        For more information please visit
        https://graphviz.readthedocs.io/en/stable/api.html#digraph.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance to be converted.
    tree_index : int, optional (default=0)
        The index of a target tree to convert.
    show_info : list of strings or None, optional (default=None)
        What information should be shown in nodes.

            - ``'split_gain'`` : gain from adding this split to the model
            - ``'internal_value'`` : raw predicted value that would be produced by this node if it was a leaf node
            - ``'internal_count'`` : number of records from the training data that fall into this non-leaf node
            - ``'internal_weight'`` : total weight of all nodes that fall into this non-leaf node
            - ``'leaf_count'`` : number of records from the training data that fall into this leaf node
            - ``'leaf_weight'`` : total weight (sum of hessian) of all observations that fall into this leaf node
            - ``'data_percentage'`` : percentage of training data that fall into this node
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    orientation : string, optional (default='horizontal')
        Orientation of the tree.
        Can be 'horizontal' or 'vertical'.
    **kwargs
        Other parameters passed to ``Digraph`` constructor.
        Check https://graphviz.readthedocs.io/en/stable/api.html#digraph for the full list of supported parameters.

    Returns
    -------
    graph : graphviz.Digraph
        The digraph representation of specified tree.
    """
    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    model = booster.dump_model()
    tree_infos = model['tree_info']
    if 'feature_names' in model:
        feature_names = model['feature_names']
    else:
        feature_names = None

    if tree_index < len(tree_infos):
        tree_info = tree_infos[tree_index]
    else:
        raise IndexError('tree_index is out of range.')

    if show_info is None:
        show_info = []

    graph = _to_graphviz(tree_info, show_info, feature_names, precision,
                         orientation, **kwargs)

    return graph


from io import BytesIO


def plot_tree(booster, ax=None, tree_index=0, figsize=None, dpi=None,
              show_info=None, precision=3, orientation='horizontal', **kwargs):
    """Plot specified tree.

    Each node in the graph represents a node in the tree.

    Non-leaf nodes have labels like ``Column_10 <= 875.9``, which means
    "this node splits on the feature named "Column_10", with threshold 875.9".

    Leaf nodes have labels like ``leaf 2: 0.422``, which means "this node is a
    leaf node, and the predicted value for records that fall into this node
    is 0.422". The number (``2``) is an internal unique identifier and doesn't
    have any special meaning.

    .. note::

        It is preferable to use ``create_tree_digraph()`` because of its lossless quality
        and returned objects can be also rendered and displayed directly inside a Jupyter notebook.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance to be plotted.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    tree_index : int, optional (default=0)
        The index of a target tree to plot.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    show_info : list of strings or None, optional (default=None)
        What information should be shown in nodes.

            - ``'split_gain'`` : gain from adding this split to the model
            - ``'internal_value'`` : raw predicted value that would be produced by this node if it was a leaf node
            - ``'internal_count'`` : number of records from the training data that fall into this non-leaf node
            - ``'internal_weight'`` : total weight of all nodes that fall into this non-leaf node
            - ``'leaf_count'`` : number of records from the training data that fall into this leaf node
            - ``'leaf_weight'`` : total weight (sum of hessian) of all observations that fall into this leaf node
            - ``'data_percentage'`` : percentage of training data that fall into this node
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    orientation : string, optional (default='horizontal')
        Orientation of the tree.
        Can be 'horizontal' or 'vertical'.
    **kwargs
        Other parameters passed to ``Digraph`` constructor.
        Check https://graphviz.readthedocs.io/en/stable/api.html#digraph for the full list of supported parameters.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with single tree.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    graph = create_tree_digraph(booster=booster, tree_index=tree_index,
                                show_info=show_info, precision=precision,
                                orientation=orientation, **kwargs)

    s = BytesIO()
    s.write(graph.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax
