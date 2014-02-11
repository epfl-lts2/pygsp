class Filter:

    """Generic graph filter and approximation by Chebyshev polynomial."""

    def __init__(self, g_function):
        """The function characterizing the filter. Should be defined in [0,1]"""
        self.g = g_function
        self.Cheb_coef = None

    def compute_Cheb_coef(self, n):
        """TODO comp. Chebyshev coefficients given g and approximation order"""
        self.Cheb_coef = np.arange(n)  # of course this has to change!





def apply_fiter(graph_props, filter, s):
    """ Apply arbitrary filter to a signal residing on nodes of a graph.

    Inputs:

        - graph_props is the graph properties (object of SpectralProp class)
        - filter: object of Filter class. It is applied in the graph 
        frequency domain of the specified graph type.
        - s is the signal we want to filter

    Output: 

        - Filtered signal
    """
    




