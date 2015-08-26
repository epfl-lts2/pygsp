# -*- coding: utf-8 -*-

from pygsp.utils import build_logger
from pygsp.graphs import gutils

import numpy as np
from scipy import sparse
from copy import deepcopy


class Graph(object):
    r"""
    The main graph object

    It is used to initialize by default every missing field of the subgraphs
    It can also be used alone to initialize customs graphs.

    Parameters
    ----------
    W : sparse matrix or ndarray
        Weights matrix (default is empty)
    A : sparse adjacency matrix
        default is constructed with W
    N : int
        Number of nodes. Default is the lenght of the first dimension of W.
    d : float
        Degree of the vectors
    Ne : int
        Edge numbers
    gtype : string
        Graph type (default is "unknown")
    directed : bool
        Whether the graph is directed
        (default depending of the previous values)
    lap_type : string
        Laplacian type (default = 'combinatorial')
    L : Ndarray
        Laplacian matrix
    coords : ndarray
        Coordinates of the vertices (default = np.array([0, 0]))
    plotting : Dict
        Dictionnary containing the plotting parameters

    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> W = np.arange(4).reshape(2, 2)
    >>> G = graphs.Graph(W)

    """

    # All the parameters that needs calculation to be set
    # or not needed are set to None
    def __init__(self, W=None, A=None, N=None, d=None, Ne=None,
                 gtype='unknown', directed=None, coords=None,
                 lap_type='combinatorial', L=None,
                 plotting={}, **kwargs):

        self.logger = build_logger(__name__)

        self.gtype = gtype
        self.lap_type = lap_type

        if W is not None:
            self.W = sparse.lil_matrix(W)
        else:
            self.W = sparse.lil_matrix(0)
        if A is not None:
            self.A = A
        else:
            self.A = sparse.lil_matrix(W > 0)
        if N is not None:
            self.N = N
        else:
            self.N = np.shape(self.W)[0]
        if d is not None:
            self.d = d
        else:
            self.d = self.W.sum()
        if Ne is not None:
            self.Ne = Ne
        else:
            self.Ne = self.W.nnz
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.zeros((self.N, 2))
        if directed:
            self.directed = directed
        else:
            self.directed = gutils.is_directed(self.W)
        if L is not None:
            self.L = L
        else:
            self.L = gutils.create_laplacian(self)

        # Plotting default parameters
        self.plotting = {}
        if 'edge_width' in plotting:
            self.plotting['edge_width'] = plotting['edge_width']
        else:
            self.plotting['edge_width'] = 1
        if 'edge_color' in plotting:
            self.plotting['edge_color'] = plotting['edge_color']
        else:
            self.plotting['edge_color'] = np.array([255, 88, 41])/255.
        if 'edge_style' in plotting:
            self.plotting['edge_style'] = plotting['edge_style']
        else:
            self.plotting['edge_style'] = '-'
        if 'vertex_size' in plotting:
            self.plotting['vertex_size'] = plotting['vertex_size']
        else:
            self.plotting['vertex_size'] = 5
        if 'vertex_color' in plotting:
            self.plotting['vertex_color'] = plotting['vertex_color']
        else:
            self.plotting['vertex_color'] = 'b'

    def update_graph_attr(self, *args, **kwargs):
        r"""
        update_graph_attr will recompute the some attribute of the graph:

        Parameters
        ----------
        args: list of string
            the arguments, that will be not changed and not re-compute.
        kwargs: Dictionnary
            The arguments with their new value.

        Return
        ------
        The same Graph with some updated values.

        Note
        ----
        This method is usefull if you want to give a new weight matrix
        (W) and compute the adjacency matrix (A) and more again.
        The valid attributes are ['W', 'A', 'N', 'd', 'Ne', 'gtype',
        'directed', 'coords', 'lap_type', 'L', 'plotting']

        Examples
        --------
        >>> from pygsp import graphs
        >>> G = graphs.Ring(N=10)
        >>> newW = G.W
        >>> newW[1] = 1
        >>> G.update_graph_attr('N', 'd', W=newW)

        Updates all attributes of G excepted 'N' and 'd'
        """
        graph_attr = {}
        valid_attributes = ['W', 'A', 'N', 'd', 'Ne', 'gtype', 'directed',
                            'coords', 'lap_type', 'L', 'plotting']

        for i in args:
            if i in valid_attributes:
                graph_attr[i] = getattr(self, i)
            else:
                self.logger.warning('Your attribute {} do not figure is the valid_attributes who are {}'.format(i, valid_attributes))

        for i in kwargs:
            if i in valid_attributes:
                if i in graph_attr:
                    self.logger.info('You already give this attribute in the args. Therefore, it will not be recaculate.')
                else:
                    graph_attr[i] = kwargs[i]
            else:
                self.logger.warning('Your attribute {} do not figure is the valid_attributes who are {}'.format(i, valid_attributes))

        from nngraphs import NNGraph
        if isinstance(self, NNGraph):
            super(NNGraph, self).__init__(**graph_attr)

        else:
            super(type(self), self).__init__(**graph_attr)

    def deep_copy_graph(self):
        r"""
        Creates a deepcopy of a graph with all the attributes.

        Exemples
        --------
        >>> from pygsp import graphs
        >>> G = graphs.Logo()
        >>> Gcopy = G.deep_copy_graph()

        """
        return deepcopy(self)

    def copy_graph_attributes(self, Gn, ctype=True):
        r"""
        Copy_graph_attributes copies some parameters of the graph into
        a given one

        Parameters
        ----------:
        G : Graph structure
        ctype : bool
            Flag to select what to copy (Default is True)
        Gn : Graph structure
            The graph where the parameters will be copied

        Returns
        -------
        Gn : Partial graph structure

        Examples
        --------
        >>> from pygsp import graphs
        >>> Torus = graphs.Torus()
        >>> G = graphs.TwoMoons()
        >>> G.copy_graph_attributes(ctype=False, Gn=Torus);

        """
        if hasattr(self, 'plotting'):
            Gn.plotting = self.plotting

        if ctype:
            if hasattr(self, 'coords'):
                Gn.coords = self.coords
        else:
            if hasattr(Gn.plotting, 'limits'):
                del Gn.plotting['limits']

        if hasattr(self, 'lap_type'):
            Gn.lap_type = self.lap_type
            Gn.L = gutils.create_laplacian(Gn)

    def separate_graph(self):
        r"""
        """
        raise NotImplementedError("Not implemented yet")

    def subgraph(self, c):
        r"""
        Create a subgraph from G.

        Parameters
        ----------
        G : graph
            Original graph
        c : int
            Node to keep

        Returns
        -------
        subG : graph
            Subgraph

        Examples
        --------
        >>> from pygsp import graphs
        >>> import numpy as np
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> c = 3
        >>> subG = graphs.Graph.subgraph(G, c)

        This function create a subgraph from G taking only the node in c.

        """

        sub_G = self
        sub_G.W = self.W[c, c]
        try:
            sub_G.N = len(c)
        except TypeError:
            sub_G.N = 1

        sub_G.gtype = "sub-" + self.gtype

        return sub_G
