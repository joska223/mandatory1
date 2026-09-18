from Wave2D import Wave2D, Wave2D_Neumann

a = Wave2D()
ani = a.animate(N = 30, Nt = 100)
ani.save("wave2d.gif", writer="pillow", fps=20)

b = Wave2D_Neumann()
ani = b.animate(N = 30, Nt = 100)
ani.save("wave2d_Neumann.gif", writer="pillow", fps=20)