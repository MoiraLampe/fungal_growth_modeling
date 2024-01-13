#!/usr/bin/env python3
# coding: utf-8

"""
model.py

"""

# get_ipython().run_line_magic('matplotlib', 'inline')

from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
import threading
import datetime
import timeit
import scipy
import tqdm
import time
import sys

# #### Vesicules
# 

# δω/δt - v δω/δx - Dδ²ω/δx² = γCin

# \begin{equation}
#     \frac{\delta \omega}{\delta t} - v\frac{\delta \omega}{\delta x} - D_{omega} \frac{\delta^{2}omega(x, t)}{\delta x^{2}} = \gamma C_{in}
# \end{equation}

# δω/δt = v δω/δx + Dδ²ω/δx² + γCin

# \begin{equation}
#     \frac{\delta \omega}{\delta t} = v\frac{\delta \omega}{\delta x} - D_{omega} \frac{\delta^{2}omega(x, t)}{\delta x^{2}} + \gamma C_{in}
# \end{equation}

# δω/δt = v (ωᵢ₋₁ - ωᵢ)/δx + D Laplacian(ω) + γCin

# \begin{equation}
#     \omega_{j}^{i+1} = \omega_{j}^{i} + dt * ( - v \frac{\omega_{j+1}^{i}-\omega_{j}^{i}}{dx} + \gamma C_{in})
# \end{equation}

# ##### Nutrients
# 
# \begin{equation}
# \label{eq:head_equation_1d}
# 	\delta_{t}C_{in}(x, t) = D \frac{\delta^{2}C_{in}}{\delta x^{2}}(x, t) + J_{C_{in}}(x,t) - (\alpha \gamma + m) C_{in}
# \end{equation}

# \begin{equation}
#     D2 = \begin{bmatrix}
# 			b & c & 0 & \cdots & 0\\
# 			a & b & c & 0 & \vdots\\
# 			0 & \ddots & \ddots & \ddots & 0\\
# 			\vdots & 0 & a & b & c\\
# 			0 & \cdots & 0 & a & b
#     \end{bmatrix}
# \end{equation}

# \begin{equation}
# \label{eq:a}
# 	A = \frac{\alpha \Delta t}{\Delta x^{2}}D2
# \end{equation}

# \begin{equation}
# \label{eq:crank_nicolson_method}
# 	\frac{T_{i}^{n+1}-T_{i}^n}{\Delta t} = \frac{\alpha}{2} \left(\frac{T_{i-1}^{n+1}-2T_{i}^{n+1}+T_{i+1}^{n+1}}{\Delta x^{2}} + \frac{T_{i-1}^{n}-2T_{i}^{n}+T_{i+1}^{n}}{\Delta x^{2}}\right)
# \end{equation}

# \begin{equation}
# \label{eq:crank_nicolson_method_matrix}
# 	(I-A_{cn})T^{n+1} = (I + A_{cn})T^{n} \Leftrightarrow T^{n+1} = (I-A_{cn})^{-1}(I + A_{cn})T^{n}
# \end{equation}

def d2_mat_dirichlet(nx, dx):
	diagonals = [[1.], [-2.], [1.]]
	offsets = [-1, 0, 1]
	d2mat = scipy.sparse.diags(diagonals, offsets, shape=(nx-2,nx-2)).toarray()
	return d2mat / dx**2

def laplacian(Z, dx):
	Zleft = Z[0:-2]
	Zright = Z[2:]
	Zcenter = Z[1:-1]
	return ((Zleft + Zright) - 2 * Zcenter) / dx**2

def system_pde(t, y, *args):
	mask, ind, bnd, n, v, D_c, D_omega, gamma, alpha, m, dx, dt, = args
	omega_ind, C_ind, JC_ind, L_ind = ind
	omega_bnd, C_bnd, JC_bnd = bnd

	omega 	= y[omega_ind]
	C 		= y[C_ind]
	JC 		= y[JC_ind]
	L 		= y[L_ind]
	
	res 	= np.empty(y.shape)

	# vesicle concentration
	res[omega_ind[1:-n-1]] = v * (omega[2:-n] - omega[:-n-2]) / dx + D_omega * laplacian(omega[0:-n], dx) + gamma * C[1:-n-1]
	# vesicle concentration boundary condition
	res[omega_ind] = np.where(mask, res[omega_ind], omega_bnd)
	res[omega_ind[-n]] = L
	# nutrient concentration
	res[C_ind[1:-n-1]] = D_c * laplacian(C[0:-n], dx)
	# res[C_ind[1:-1]] = np.dot(M3, C[1:-1]) + np.dot(M1, (JC[1:-1] - (alpha * gamma + m) * C[1:-1]) * dt)
	# nutrient concentration boundary condition
	res[C_ind] = np.where(mask, res[C_ind], C_bnd)
	# nutrient flux
	res[JC_ind[1:-n-1]] = -1e-2 * JC[1:-n-1]
	# nutrient flux boundary condition
	res[JC_ind] = np.where(mask, res[JC_ind], JC_bnd)
	# length
	res[L_ind] = 1 / (1 + np.exp(-omega[-n])) 
	
	return res

def print_sol(y, indices):
	omega_ind, C_ind, JC_ind, L_ind = indices
	print("shape")
	print(y.shape)
	print("omega")
	print(y[omega_ind])
	print("C")
	print(y[C_ind])
	print("JC")
	print(y[JC_ind])
	print("L")
	print(y[L_ind])

class Timer:
	"""
		Timer class 
		Displays elapsed time
	"""

	def __init__(self, round_ndigits: int = 0):
		self._round_ndigits = round_ndigits
		self._start_time = timeit.default_timer()

	def __call__(self) -> float:
		return timeit.default_timer() - self._start_time

	def __str__(self) -> str:
		return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))

def animate():
	timer = Timer()
	symbols = ['|', '/', '-', '\\']
	for c in itertools.cycle(symbols):
		if done:
			break
		sys.stdout.write(f'\rLoading {c} - {timer}')
		sys.stdout.flush()
		time.sleep(1 / len(symbols))
	sys.stdout.write(f'\rDone! Took {timer} sec')


if __name__ == '__main__':
	# ##### Parameters

	# minutes
	# µ meters
	# gL^-1

	# dimension
	dim 	= 1
	# diffusion coefficient
	D_c 	= 2.48e-4
	# diffusion coefficient
	D_omega = 2.48e-4
	# length
	L 		= 1
	# points
	P 		= int(1e2)
	# delta length
	dx 		= L / P
	# time span
	t_span 	= (0., 1.)
	# delta time
	dt = (0.5 * dx**2) / D_c
	# vesicle transport speed (µm / min)
	v 		= 12 # 12.3
	# vesicle production rate
	gamma 	= 2e-8
	# convertion factor (nutrients to vesicles)
	alpha 	= 1e1
	# nutrient used for maintenance
	m 		= 1.8e-3

	print(f"dt: {dt:.6f}")
	print(f"dx: {dx:.6f}")

	# grid points X axis
	X 	= np.linspace(0, L, P)
	# number of discritized space 
	nx 	= len(X)

	# initial number of tanks
	n_tanks 	= 5
	# iteration counter
	count 		= 0
	# start time
	t 			= t_span[0]

	# ##### Simulation

	indices 	= np.indices(X.shape)

	# ###### initial & boundary condition & indicies
	# vesicle concentration
	omega0 		= np.zeros(nx)
	omega_bnd 	= np.zeros(nx)
	omega_ind 	= np.arange(0, nx)
	# nutrient concentration
	C0 			= np.zeros(nx)
	C_bnd 		= np.zeros(nx)
	C_ind 		= np.arange(nx, 2*nx)
	# nutrient flux
	JC0 		= np.zeros(nx)
	JC0[1:-(P-n_tanks)] = 1 # 1e-6
	JC_bnd 		= np.zeros(nx)
	JC_ind 		= np.arange(2*nx, 3*nx)
	# length
	L0 			= np.full((1), n_tanks * dx)
	L_ind 		= np.arange(3*nx, 3*nx+1)

	y0 			= np.array([*omega0, *C0, *JC0, *L0])

	times 		= np.array([t])
	solutions 	= np.array([y0])

	# progress bars
	with \
		tqdm.tqdm(
			position=0, 
			initial=n_tanks,
			desc="TANKS",
		) as tanks_pbar, \
		tqdm.tqdm(
			position=1, 
			desc="COUNT",
		) as count_pbar \
	:
		while True:
			# get boundary mask for square matrix
			mask = np.logical_and.reduce(
				np.ma.masked_inside(
					indices, 
					indices.min()+1, 
					n_tanks-1
				).astype(bool).mask
			)

			# solve system of ODE
			sol = scipy.integrate.solve_ivp(
				system_pde, 
				[t, t + dt], 
				solutions[-1], 
				args=(
					mask,
					(omega_ind, C_ind, JC_ind, L_ind),
					(omega_bnd, C_bnd, JC_bnd),
					P - n_tanks, 
					v,
					D_c,
					D_omega,
					gamma,
					alpha,
					m,
					dx, 
					dt,
				),
				method='RK23',
				# t_eval=dt
			)

			# current length
			nl = n_tanks * dx
			# get firt index where the length is greater than the current length
			i_end = np.where(sol.y[-1] >= nl)[0][0] - 1
			# indicies from 1 to next length
			i = np.arange(1, i_end if i_end != -1 else sol.y.shape[1])
			# update current time
			t = sol.t[i_end-1] if i_end != -1 else sol.t[i_end]
			solutions = np.concatenate((solutions, sol.y.T[i]), axis=0)
			times = np.concatenate((times, sol.t[i]), axis=0)

			# check if we added a tank
			if sol.y[-1][-1] >= nl:
				# update nutrient flux so that the new tank has an inital value of 1
				solutions[-1, JC_ind[n_tanks]] = 1
				# update number of tanks
				n_tanks += 1
				# publish update to progress bar
				tanks_pbar.update(1)

			# counter for number of times we call solve_ivp
			count += 1
			# publish update to progress bar
			count_pbar.update(1)

			if n_tanks == P or count > 1e9: 
				break

	# ###### Solutions Data Visualisation ######

	done = False

	thread = threading.Thread(target=animate)
	thread.start()

	# solutions = np.fmax(solutions, 0)

	C 		= solutions[:, C_ind]
	omega 	= solutions[:, omega_ind]
	L 		= solutions[:, L_ind]
	JC 		= solutions[:, JC_ind]

	C 		= C[::10]
	omega 	= omega[::10]
	L 		= L[::10]
	JC 		= JC[::10]

	times = times[::10]

	# ###### Nutrient Concentration Graph ######

	fig = plt.figure(figsize=(10, 10))
	plt.suptitle(
		f"dt = ${dt}$" + \
		f", dx = ${dx}$" + \
		f", tanks = ${n_tanks}$" + \
		f", D_c = ${D_c}$" + \
		f", D_omega = ${D_omega}$" + \
		f"\n" + \
		f", L = ${P*dx}$" + \
		f", P = ${P}$" + \
		f", v = ${v}$" + \
		f", gamma = ${gamma}$" + \
		f", alpha = ${alpha}$" + \
		f", m = ${m}$", 
		fontsize=14
	)

	norm = plt.Normalize(C.min(), C.max())
	colors = mpl.cm.viridis(norm(C))
	rcount, ccount, _ = colors.shape

	ax = fig.add_subplot(221, projection='3d')

	data = (*np.meshgrid(X, times), C)

	ax.plot_surface(
		*data, 
		rcount=rcount, 
		ccount=ccount, 
		facecolors=colors, 
		shade=False
	)
	ax.set_facecolor((0,0,0,0))

	ax.view_init(elev=30, azim=-45, roll=1)

	ax.set_xlabel("$x$")
	ax.set_ylabel("Time ($min$)")
	ax.set_zlabel("Nutrient Concentration ($gL^{-1}$)")

	ax.set_title('Nutrient Concentration')

	# ###### Vesicle Concentration Graph ######

	norm = plt.Normalize(omega.min(), omega.max())
	colors = mpl.cm.viridis(norm(omega))
	rcount, ccount, _ = colors.shape

	ax = fig.add_subplot(222, projection='3d')

	data = (*np.meshgrid(X, times), omega)

	ax.plot_surface(
		*data, 
		rcount=rcount, 
		ccount=ccount, 
		facecolors=colors, 
		shade=False
	)
	ax.set_facecolor((0,0,0,0))

	ax.view_init(elev=30, azim=-45, roll=1)

	ax.set_xlabel("$x$")
	ax.set_ylabel("Time ($min$)")
	ax.set_zlabel("Vesicle Concentration ($gL^{-1}$)")

	ax.set_title('Vesicle Concentration')

	# ###### Nutrient Flux Concentration Graph ######

	norm = plt.Normalize(JC.min(), JC.max())
	colors = mpl.cm.viridis(norm(JC))
	rcount, ccount, _ = colors.shape

	ax = fig.add_subplot(223, projection='3d')

	data = (*np.meshgrid(X, times), JC)

	ax.plot_surface(
		*data, 
		rcount=rcount, 
		ccount=ccount, 
		facecolors=colors, 
		shade=False
	)
	ax.set_facecolor((0,0,0,0))

	ax.view_init(elev=30, azim=80, roll=1)

	ax.set_xlabel("$x$")
	ax.set_ylabel("Time ($min$)")
	ax.set_zlabel("Nutrient Flux Concentration ($gL^{-1}$)")

	ax.set_title('Nutrient Flux Concentration')

	# # ###### Length Graph ######

	norm = plt.Normalize(L.min(), L.max())
	colors = mpl.cm.viridis(norm(L))
	rcount, ccount, _ = colors.shape

	ax = fig.add_subplot(224, projection='3d')

	data = (*np.meshgrid(X, times), L)

	ax.plot_surface(
		*data, 
		rcount=rcount, 
		ccount=ccount, 
		facecolors=colors, 
		shade=False
	)
	ax.set_facecolor((0,0,0,0))

	ax.view_init(elev=30, azim=-10, roll=1)

	ax.set_xlabel("$x$")
	ax.set_ylabel("Time ($min$)")
	ax.set_zlabel("Length ($μm$)")

	ax.set_title('Length')

	# plt.show()
	plt.savefig('../figures/Figure_.png')

	done = True

	thread.join()




