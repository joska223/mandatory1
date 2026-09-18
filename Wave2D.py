import numpy as np
import sympy as sp
from scipy import sparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""

        xi = np.linspace(0,1,N+1)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """

        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2

        return D.tocsr()

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c*np.pi*np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        xij, yij = self.create_mesh(N)
        np_ue = sp.lambdify((x,y,t), self.ue(mx, my), modules = "numpy")
        u0 = np_ue(xij, yij, 0)

        return u0.ravel()


    @property
    def dt(self) -> float:
        """Return the time step"""
        h = 1/self.N
        return self.cfl*h/self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = int(np.sqrt(len(u[0])))-1
        t_steps = len(u)
        error = np.zeros(t_steps)

        xij, yij = self.create_mesh(N)

        for i in range(t_steps):
            ut = u[i].reshape(N+1,N+1)

            np_ue = sp.lambdify((x,y,t), self.ue(self.mx, self.my), modules = "numpy")
            u_e = np_ue(xij, yij, i*self.dt)

            error[i] = (1/N)*np.sqrt(np.sum((ut-u_e)**2))
        return error


    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        U = u.reshape(self.N+1, self.N+1)

        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0

        return U.ravel()

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        self.mx = mx
        self.my = my
        self.N = N
        self.c = c
        self.cfl = cfl

        data = {}

        u = self.initialize(N, mx, my)

        D2 = self.D2(N)

        I = sparse.eye(N+1, format="csr")
        A = (
            sparse.kron(D2, I, format="csr")
            + sparse.kron(I, D2, format="csr")
        )

        up1 = (0.5)*cfl**2*A@u+u
        up1 = self.apply_bcs(up1)

        if store_data > 0 or store_data == -1:
            data[0] = u
            data[1] = up1

        for i in range(2,Nt+1):
            um1 = u
            u = up1
            up1 = cfl**2*A@u+2*u-um1
            up1 = self.apply_bcs(up1)
            if store_data > 0 or store_data == -1:
                data[i] = up1

        if store_data > 0:
            return data

        if store_data == -1:
            h = 1/N
            error = self.l2_error(data, Nt*self.dt)

            return h, error
        
                


    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)

    def animate(
        self,
        N: int = 10,
        Nt: int = 30,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 2,
        my: int = 2,
        ):
        data = self(
            N=N,
            Nt=Nt,
            cfl=cfl,
            c=c,
            mx=mx,
            my=my,
            store_data=1,
        )

        X, Y = self.create_mesh(N)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Initial surface
        Z = data[0].reshape(N + 1, N + 1)

        surf = ax.plot_surface(
            X, Y, Z,
            cmap="coolwarm",
            linewidth=0,
            antialiased=False,
            vmin=-1,
            vmax=1,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_zlim(-1, 1)

        def update(frame):
            nonlocal surf

            # Remove previous surface
            surf.remove()

            # Get solution at this timestep
            Z = data[frame].reshape(N + 1, N + 1)

            # Draw new surface
            surf = ax.plot_surface(
                X, Y, Z,
                cmap="coolwarm",
                linewidth=0,
                antialiased=False,
                vmin=-1,
                vmax=1,
            )

            ax.set_title(f"t = {frame * self.dt:.3f}")

            return (surf,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(Nt + 1),
            interval=50,
            blit=False,
        )

        return ani


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D2 = super().D2(N)
        D2[0,:4] = np.array([-2, 2, 0, 0])
        D2[-1,-4:] = np.array([0, 0, 2, -2])

        return D2

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        return u


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    tol = 10**(-12)
    sol = Wave2D()
    _, E, _ = sol.convergence_rates(mx = 3, my = 3, cfl = 1/np.sqrt(2))
    assert(E[-1]) < tol

    solN = Wave2D_Neumann()
    _, E_N, _ = solN.convergence_rates(mx = 3, my = 3, cfl = 1/np.sqrt(2))
    assert(E_N[-1]) < tol


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")

