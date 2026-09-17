import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2

        return D.tocsr()

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """

        xij, yij = self.create_mesh(N)

        mesh_f = self.meshfunction(f, xij, yij).ravel()

        bnds = self.get_boundary_indices(N)

        ue_np = sp.lambdify((x,y), ue, modules = "numpy")
        BC = ue_np(xij, yij).ravel()[bnds]

        b = mesh_f.ravel()
        b[bnds] = BC
        

        D = self.laplace(N)    
        h = self.p.L/N

        D2x = (1./h**2)*D
        D2y = (1./h**2)*D
        I = sparse.eye(N + 1, format="csr")

        A = (
            sparse.kron(D2x, I, format="csr")
            + sparse.kron(I, D2y, format="csr")
        )

        A = A.tolil()

        for i in bnds:
            A[i, :] = 0
            A[i, i] = 1.0

        A = A.tocsr()

        return A, b


    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        ue_np = sp.lambdify((x,y), u, modules = "numpy")
        b = ue_np(xij, yij)
        return b.ravel()


    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((N+1, N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error
        """
        N = len(u[0,:])-1
        h = self.p.L / N
        ue_np = sp.lambdify((x,y), ue, modules = "numpy")
        xij, yij = self.create_mesh(N)
        b = ue_np(xij, yij)
        a = np.sum((u-b)**2)
        return np.sqrt((h**2)*a)


    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        print("  assembling", N)
        A, b = self.assemble(
            N,
            sp.diff(ue, x, 2) + sp.diff(ue, y, 2),
            ue
        )

        print("  assembled", N, "nnz =", A.nnz)
        print("  solving", N)

        u = sparse_linalg.spsolve(A, b.ravel())

        print("  solved", N)
        return u.reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        
        N = len(U[0,:])-1
        L = self.p.L 

        #find the closest index to the requested point
        x_pos = (x/L)*N
        y_pos = (y/L)*N
        
        x_idx = int(np.floor((x/L)*N))
        y_idx = int(np.floor((y/L)*N))
     
        x_space = x_pos-x_idx
        y_space = y_pos-y_idx

        ans =   (1-x_space)*(1-y_space)*U[x_idx, y_idx]+\
            (1-x_space)*y_space*U[x_idx, y_idx+1]+\
            x_space*(1-y_space)*U[x_idx+1, y_idx]+\
        x_space*y_space*U[x_idx+1, y_idx+1]
    
        return ans


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
