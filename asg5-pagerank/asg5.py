import numpy as np
from numpy import linalg as la


np.set_printoptions(precision=3)



class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        Labellist (list(str)): List of labels for the n nodes in the 
            graph.
        A_hat ((n,n) ndarray): The adjacency matrix of a directed graph,
            where A_hat has been calculated from matrix A.
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].                

        Examples
        ========
        >>> A = np.array([[0, 0, 0, 0],[1, 0, 1, 0],[1, 0, 0, 1],[1, 0, 1, 0]])
        >>> G = DiGraph(A, labels=['a','b','c','d'])
        >>> G.A_hat
        array([[0.   , 0.25 , 0.   , 0.   ],
               [0.333, 0.25 , 0.5  , 0.   ],
               [0.333, 0.25 , 0.   , 1.   ],
               [0.333, 0.25 , 0.5  , 0.   ]])
        >>> steady_state_1 = G.linsolve()
        >>> { k: round(steady_state_1[k],3) for k in steady_state_1}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_2 = G.eigensolve()
        >>> { k: round(steady_state_2[k],3) for k in steady_state_2}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> steady_state_3 = G.itersolve()
        >>> { k: round(steady_state_3[k],3) for k in steady_state_3}
        {'a': 0.096, 'b': 0.274, 'c': 0.356, 'd': 0.274}
        >>> get_ranks(steady_state_3)
        ['c', 'b', 'd', 'a']
        """
        A = A * 1.0 #security to convert elements in the matrix to floats, since it's required for calculating A_hat.
        
        #check if there's sinks, and if there is, modify the matrix to have none.
        i = 0 #iterator to remember which column we're on.
        for row in A.transpose():
            i = i + 1
            if row.sum() == 0:
                A[:,i-1] = 1
        i = 0
        #Now calculating A_hat
        for row in A.transpose():
            i = i + 1
            A[:,i-1] = A[:,i-1] / row.sum()
        self.A_hat = A #saving A_hat as attribute   
        
        if labels is None:
            labellist = list()
            labellist = range(0, np.size(A,1))
            self.labellist = [str(n) for n in labellist]

        if labels is not None:
            if len(labels) > np.size(A,1):
                raise ValueError('Amount of labels exceeds number of nodes!')
            else:
                self.labellist = labels

    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        n = np.size(self.A_hat,1)
        results = np.linalg.solve(np.eye(n, n) - np.dot(epsilon, self.A_hat),(1 - epsilon) * (np.array([1] * n) / n))
        dict = {}
        for i in range(n):
            dict[self.labellist[i]] = results[i]
        return dict

    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        n = np.size(self.A_hat,1)
        A_flat = epsilon * self.A_hat + (1 - epsilon) * (np.ones((n,n)) / n)
        w,v = np.linalg.eig(A_flat)
        results = abs(np.real(v[:n,0]) / np.linalg.norm(v[:n,0],1))

        dict = {}
        for i in range(n):
            dict[self.labellist[i]] = results[i]
        return dict

    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        raise NotImplementedError("Problem 2 Incomplete")

def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """    
    raise NotImplementedError("Problem 3 Incomplete")


# Task 2
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.

    Examples
    ========
    >>> print(rank_websites()[0:5])
    ['98595', '32791', '28392', '77323', '92715']
    """
    raise NotImplementedError("Task 2 Incomplete")


# Task 3
def rank_uefa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.

    Examples
    ========
    >>> rank_uefa_teams("psh-uefa-2018-2019.csv",0.85)[0:5]
    ['Liverpool', 'Ath Madrid', 'Paris SG', 'Genk', 'Barcelona']
    """
    raise NotImplementedError("Task 3 Incomplete")






if __name__ == "__main__":
    import doctest
    doctest.testmod()
