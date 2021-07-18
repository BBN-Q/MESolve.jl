# New version
"""
	CV_solve_time_independent(C_in::Array{T1,2},a_in::Vector{Float64},
													 h::Array{T2,2},
													 d::Array{T3,1},
													 Z::Array{T4,2},
													 t0::AbstractFloat,
													 tf::AbstractFloat; 
													 tstep::AbstractFloat=0.,
													 tols::Vector{Float64}=[1e-6,1e-3],
													 alg = Tsit5()) where {T1 <: Number, T2 <: Number,  T3 <: Number, T4 <: Number}

Time independent solver for N-mode Gaussian continuous variable systems 
using the covariance matrix approach. Defined in terms of the canonical 
X and P quadratures for each, the covariance matrix elements are defined by
C_ab = <cc^⊺> - <c><c^⊺> where c^⊺ = [X_1,P_1,X_2,P_2,...,X_N,P_N] with
X = (a + a^†)/√2, P = -i(a - a^†)/√2 where "a" is the lowering operator 
of a harmonic oscilaltor. Note that c is a column vector and c^⊺ is its 
transpose, a row vector.

The total Hamiltonian is H = H_quad + H_lin = 1/2*(c^⊺*h*c) + d^⊺*c

## args
* C_in:  Array 2N × 2N, initial condition for the covariance matrix
* a_in:  vector 2N × 1, initial condition for the average value vector
* h:     2N × 2N matrix, describing the quadratic part of the Hamiltonian 
	 	 evolution of the system in the X_i, P_i basis, such that H_quad = 1/2*(c^⊺*h*c)
* d:     2N × 1 vector, describing the linear part of the Hamiltonian 
	     evolution of the system in the X_i, P_i basis, such that H_lin = d^⊺*c
* Z:     Array 2N × 2N, describes the dissipative evolution of the system. 
	     Given the master equation d/dt ρ = ∑_k γ_k*D[L_k]ρ with D[L_k]ρ the 
	     standard dissipator for the Lindblad operator L_k, we describe each 
	     Lindblad operator in the X_i, P_i basis as L_k = R_k^⊺*c Then
	     Z = ∑_k γ_k*conj(R_k)*R_k^⊺
* t0:    float, start time
* tf:    float, end time
* tstep: float, time steps at which output data should be saved
* tols:  2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
* alg:   function, algorithm from DifferentialEquations for the solver to use, default is Tsit5

## returns
* sol_C.t, sol_C.u, sol_a.t, sol_a.u
"""
function CV_solve_time_independent(C_in::Array{T1,2},a_in::Vector{Float64},
													 h::Array{T2,2},
													 d::Array{T3,1},
													 Z::Array{T4,2},
													 t0::AbstractFloat,
													 tf::AbstractFloat; 
													 tstep::AbstractFloat=0.,
													 tols::Vector{Float64}=[1e-6,1e-3],
													 alg = Tsit5()) where {T1 <: Number, T2 <: Number,  T3 <: Number, T4 <: Number}

	C_in = convert(Array{ComplexF64,2}, C_in)
	h = convert(Array{Float64,2}, h)
	d = convert(Array{Float64,1}, d)
	Z = convert(Array{ComplexF64,2}, Z)

	if tstep < 1e-10
        tstep = (tf-t0)./100
    end
	tspan = (t0,tf)

	σ = zeros(Float64,size(C_in,1),size(C_in,2))
	for nn = 1:2:size(C_in,1)
		σ[nn:1:nn+1,nn:1:nn+1] = [0. 1.;-1. 0.]
	end

	# Covariance matrix
	M = σ*(h + imag.(Z))
	Zs = σ*Z*transpose(σ)

    dif_C(du,u,p,t) = dCV(du,u,M,Zs,t) # In place

    prob_C = ODEProblem{true}(dif_C,C_in,tspan)

    sol_C = solve(prob_C, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])

	# Average
	d = σ*d

	dif_a(du,u,p,t) = d_av(du,u,M,d,t) # In place

    prob_a = ODEProblem{true}(dif_a,a_in,tspan)

    sol_a = solve(prob_a, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])

	return sol_C.t, sol_C.u, sol_a.t, sol_a.u
end


"""
    CV_solve_time_dependent(C_in::Array{T1,2},a_in::Vector{Float64},
												   h::Array{T2,2},
												   d::Function,
												   Z::Array{T3,2},
												   t0::AbstractFloat,
												   tf::AbstractFloat; 
												   tstep::AbstractFloat=0.,
												   tols::Vector{Float64}=[1e-6,1e-3],
												   alg = Tsit5()) where {T1 <: Number, T2 <: Number,  T3 <: Number}

Time dependent (drive only) solver for N-mode Gaussian continuous variable systems using the covariance matrix approach.

args:
REQUIRED
	C_in: Array 2N × 2N, initial condition for the covariance matrix

	a_in: vector 2N × 1, initial condition for the average value vector

	h: 2N × 2N matrix, describing the quadratic part of the Hamiltonian evolution of the system in the X_i, P_i basis, such that
		H_quad = 1/2*(c^⊺*h*c)

	d: Function that returns as output a 2N × 1 vector, describing the linear part of the Hamiltonian evolution of the system in the X_i, P_i basis, such that
		H_lin = d^⊺*c

	Z: Array 2N × 2N, describes the dissipative evolution of the system. Given the master equation
	d/dt ρ = ∑_k γ_k*D[L_k]ρ
	with D[L_k]ρ the standard dissipator for the Lindblad operator L_k, we describe each Lindblad operator in the X_i, P_i basis as
	L_k = R_k^⊺*c
	Then
	Z = ∑_k γ_k*conj(R_k)*R_k^⊺

	t0: float, start time

	tf: float, end time
KEYWORD OPTIONAL
	tstep: float, time steps at which output data should be saved
	tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
	alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
"""
function CV_solve_time_dependent(C_in::Array{T1,2},a_in::Vector{Float64},
												   h::Array{T2,2},
												   d::Function,
												   Z::Array{T3,2},
												   t0::AbstractFloat,
												   tf::AbstractFloat; 
												   tstep::AbstractFloat=0.,
												   tols::Vector{Float64}=[1e-6,1e-3],
												   alg = Tsit5()) where {T1 <: Number, T2 <: Number,  T3 <: Number}

	C_in = convert(Array{ComplexF64,2}, C_in)
	h = convert(Array{Float64,2}, h)
	Z = convert(Array{ComplexF64,2}, Z)

	if tstep < 1e-10
        tstep = (tf-t0)./100
    end
	tspan = (t0,tf)

	σ = zeros(Float64,size(C_in,1),size(C_in,2))
	for nn = 1:2:size(C_in,1)
		σ[nn:1:nn+1,nn:1:nn+1] = [0. 1.;-1. 0.]
	end

	# Covariance matrix
	M = σ*(h + imag.(Z))
	Zs = σ*Z*transpose(σ)

    dif_C(du,u,p,t) = dCV(du,u,M,Zs,t) # In place

    prob_C = ODEProblem{true}(dif_C,C_in,tspan)

    sol_C = solve(prob_C, alg, saveat = tstep, dense=false, 
    										   save_everystep=false, 
    										   abstol = tols[1], 
    										   reltol = tols[2])

	# Average

	function drv(in_d::AbstractArray,t::AbstractFloat)
		in_d[:] = σ*d(t)
		nothing
	end

	drv_temp = Array{Float64}(undef,size(a_in,1))
	dif_a(du,u,p,t) = d_av(du,u,M,drv,t,drv_temp) # In place

    prob_a = ODEProblem{true}(dif_a,a_in,tspan)

    sol_a = solve(prob_a, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])

	return sol_C.t, sol_C.u, sol_a.t, sol_a.u
end

"""
Time dependent solver for N-mode Gaussian continuous variable systems using the covariance matrix approach.

Input Parameters:
REQUIRED
	C_in: Array 2N × 2N, initial condition for the covariance matrix

	a_in: vector 2N × 1, initial condition for the average value vector

	h: Function that returns as output a 2N × 2N matrix, describing the quadratic part of the Hamiltonian evolution of the system in the X_i, P_i basis, such that
		H_quad = 1/2*(c^⊺*h*c)

	d: Function that returns as output a 2N × 1 vector, describing the linear part of the Hamiltonian evolution of the system in the X_i, P_i basis, such that
		H_lin = d^⊺*c

	Z: Array 2N × 2N, describes the dissipative evolution of the system. Given the master equation
	d/dt ρ = ∑_k γ_k*D[L_k]ρ
	with D[L_k]ρ the standard dissipator for the Lindblad operator L_k, we describe each Lindblad operator in the X_i, P_i basis as
	L_k = R_k^⊺*c
	Then
	Z = ∑_k γ_k*conj(R_k)*R_k^⊺

	t0: float, start time

	tf: float, end time
KEYWORD OPTIONAL
	tstep: float, time steps at which output data should be saved
	tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
	alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
"""
function CV_solve_time_dependent(C_in::Array{T1,2},a_in::Vector{Float64},
												   h::Function,
												   d::Function,
												   Z::Array{T2,2},
												   t0::AbstractFloat,
												   tf::AbstractFloat; 
												   tstep::AbstractFloat=0.,
												   tols::Vector{Float64}=[1e-6,1e-3],
												   alg = Tsit5()) where {T1 <: Number, T2 <: Number}

	C_in = convert(Array{ComplexF64,2}, C_in)
	Z = convert(Array{ComplexF64,2}, Z)

	if tstep < 1e-10
        tstep = (tf-t0)./100
    end
	tspan = (t0,tf)

	σ = zeros(Float64,size(C_in,1),size(C_in,2))
	for nn = 1:2:size(C_in,1)
		σ[nn:1:nn+1,nn:1:nn+1] = [0. 1.;-1. 0.]
	end

	# Covariance matrix
	function M(in_M::AbstractArray,t::AbstractFloat)
		in_M[:,:] = σ*(h(t) + imag.(Z))
		nothing
	end

	Zs = σ*Z*transpose(σ)

	M_temp = Array{Float64}(undef,size(C_in,1),size(C_in,2))
    dif_C(du,u,p,t) = dCV(du,u,M,Zs,t,M_temp) # In place

    prob_C = ODEProblem{true}(dif_C,C_in,tspan)

    sol_C = solve(prob_C, alg, saveat = tstep, dense=false, 
    										   save_everystep=false, 
    										   abstol = tols[1], 
    										   reltol = tols[2])

	# Average

	function drv(in_d::AbstractArray,t::AbstractFloat)
		in_d[:] = σ*d(t)
		nothing
	end

	fill!(M_temp,0.)
	drv_temp = Array{Float64}(undef,size(a_in,1))
	dif_a(du,u,p,t) = d_av(du,u,M,drv,t,M_temp,drv_temp) # In place

    prob_a = ODEProblem{true}(dif_a,a_in,tspan)

    sol_a = solve(prob_a, alg, saveat = tstep, dense=false, 
    										   save_everystep=false, 
    										   abstol = tols[1], 
    										   reltol = tols[2])

	return sol_C.t, sol_C.u, sol_a.t, sol_a.u
end


"""
Input in the a,a^† basis

NOTE: Due to the overhead of converting from the a,a^† basis to the X, P basis, benchmarking has these implementations at least a factor of 2 slower for the time dependent solvers, and requiring at least twice the memory than the X, P basis implementations.
"""

"""
Input Parameters:
REQUIRED
	C_in: Array 2N × 2N, initial condition for the covariance matrix

	a_in: vector 2N × 1, initial condition for the average value vector

	ω: N × 1 vector of the mode frequencies

	g: N × N array of the flip-flop coupling. Only the lower triangular part is used, and g[k,j] corresponds to the rate of the term a_k^†a_j in the Hamiltonian. The diagonal should be all zeros.

	λ: N × N array of the single and two-mode squeezing rates. Only the lower triangular part (including diagonal) is used, and λ[k,j] corresponds to the rate of the term a_k^†a_j^† in the Hamiltonian.

	α: N × 1 vector of the complex drive amplitudes, where α[k] corresponds to the term a_k in the Hamiltonian

	Z: Array 2N × 2N, describes the dissipative evolution of the system. Given the master equation
	d/dt ρ = ∑_k γ_k*D[L_k]ρ
	with D[L_k]ρ the standard dissipator for the Lindblad operator L_k, we describe each Lindblad operator in the X_i, P_i basis as
	L_k = R_k^⊺*c
	Then
	Z = ∑_k γ_k*conj(R_k)*R_k^⊺

	t0: float, start time

	tf: float, end time
KEYWORD OPTIONAL
	tstep: float, time steps at which output data should be saved
	tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
	alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
"""
function CV_solve_time_independent(C_in::Array{T1,2},a_in::Vector{Float64},
													 ω::Vector{Float64},
													 g::Array{T2,2},
													 λ::Array{T3,2},
													 α::Array{T4,1},
													 Z::Array{T5,2},
													 t0::AbstractFloat,
													 tf::AbstractFloat; 
													 tstep::AbstractFloat=0.,
													 tols::Vector{Float64}=[1e-6,1e-3],
													 alg = Tsit5()) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number, T5 <: Number}

	# Conert to X,P basis
	Num = length(ω)
	h = zeros(Float64,size(C_in,1),size(C_in,2))
	d = zeros(Float64,size(C_in,1))

	for jj = 1:1:Num
		for kk = (jj+1):1:Num
			# XX
			h[2*kk-1,2*jj-1] = (real(g[kk,jj]) + real(λ[kk,jj]))

			#XY
			h[2*kk-1,2*jj] = (imag(λ[kk,jj]) - imag(g[kk,jj]))

			#YX
			h[2*kk,2*jj-1] = (imag(λ[kk,jj]) + imag(g[kk,jj]))

			#YY
			h[2*kk,2*jj] = (real(g[kk,jj]) - real(λ[kk,jj]))
		end

		#YX same
		h[2*jj,2*jj-1] = 2*imag(λ[jj,jj])
	end

	h[:,:] = h[:,:] + h[:,:]'

	h[diagind(h)] = vec([(ω+2*real(diag(λ))) (ω-2*real(diag(λ)))]')[:]

	d[:] = sqrt(2)*vec([real.(α) -imag.(α)]')[:]

	return CV_solve_time_independent(C_in,a_in,h,d,Z,t0,tf; tstep=tstep,
															tols=tols,
															alg=alg)
end

"""
Input Parameters:
REQUIRED
	C_in: Array 2N × 2N, initial condition for the covariance matrix

	a_in: vector 2N × 1, initial condition for the average value vector

	ω: N × 1 vector of the mode frequencies

	g: N × N array of the flip-flop coupling. Only the lower triangular part is used, and g[k,j] corresponds to the rate of the term a_k^†a_j in the Hamiltonian. The diagonal should be all zeros.

	λ: N × N array of the single and two-mode squeezing rates. Only the lower triangular part (including diagonal) is used, and λ[k,j] corresponds to the rate of the term a_k^†a_j^† in the Hamiltonian.

	α: Function, returns an N × 1 vector of the complex drive amplitudes, where α[k] corresponds to the term a_k in the Hamiltonian

	Z: Array 2N × 2N, describes the dissipative evolution of the system. Given the master equation
	d/dt ρ = ∑_k γ_k*D[L_k]ρ
	with D[L_k]ρ the standard dissipator for the Lindblad operator L_k, we describe each Lindblad operator in the X_i, P_i basis as
	L_k = R_k^⊺*c
	Then
	Z = ∑_k γ_k*conj(R_k)*R_k^⊺

	t0: float, start time

	tf: float, end time
KEYWORD OPTIONAL
	tstep: float, time steps at which output data should be saved
	tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
	alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
"""
function CV_solve_time_dependent(C_in::Array{T1,2},a_in::Vector{Float64},
												   ω::Vector{Float64},
												   g::Array{T2,2},
												   λ::Array{T3,2},
												   α::Function,
												   Z::Array{T4,2},
												   t0::AbstractFloat,
												   tf::AbstractFloat; 
												   tstep::AbstractFloat=0.,
												   tols::Vector{Float64}=[1e-6,1e-3],
												   alg = Tsit5()) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}

	# Conert to X,P basis
	Num = length(ω)
	h = zeros(Float64,size(C_in,1),size(C_in,2))

	for jj = 1:1:Num
		for kk = (jj+1):1:Num
			# XX
			h[2*kk-1,2*jj-1] = (real(g[kk,jj]) + real(λ[kk,jj]))

			#XY
			h[2*kk-1,2*jj] = (imag(λ[kk,jj]) - imag(g[kk,jj]))

			#YX
			h[2*kk,2*jj-1] = (imag(λ[kk,jj]) + imag(g[kk,jj]))

			#YY
			h[2*kk,2*jj] = (real(g[kk,jj]) - real(λ[kk,jj]))
		end

		#YX same
		h[2*jj,2*jj-1] = 2*imag(λ[jj,jj])
	end

	h[:,:] = h[:,:] + h[:,:]'

	h[diagind(h)] = vec([(ω+2*real(diag(λ))) (ω-2*real(diag(λ)))]')[:]

	function d(t::AbstractFloat)
		return sqrt(2)*vec([real.(α(t)) -imag.(α(t))]')[:]
	end

	return CV_solve_time_dependent(C_in,a_in,h,d,Z,t0,tf; tstep=tstep,tols=tols,alg=alg)
end

"""
The total Hamiltonian is H = ∑_k (ω_k*n_k + α_k*a_k + conj(α_k)*a_k^†) + ∑_kj (g_kj*a_k^†a_j + λ_kj*a_k^†a_j^† + h.c.)
with n = a^†a

Input Parameters:
REQUIRED
	C_in: Array 2N × 2N, initial condition for the covariance matrix

	a_in: vector 2N × 1, initial condition for the average value vector

	ω: Function, returns an N × 1 vector of the mode frequencies

	g: Function, returns an N × N array of the flip-flop coupling. Only the lower triangular part is used, and g[k,j] corresponds to the rate of the term a_k^†a_j in the Hamiltonian. The diagonal should be all zeros.

	λ: Function, returns an N × N array of the single and two-mode squeezing rates. Only the lower triangular part (including diagonal) is used, and λ[k,j] corresponds to the rate of the term a_k^†a_j^† in the Hamiltonian.

	α: Function, returns an N × 1 vector of the complex drive amplitudes, where α[k] corresponds to the term a_k in the Hamiltonian

	Z: Array 2N × 2N, describes the dissipative evolution of the system. Given the master equation
	d/dt ρ = ∑_k γ_k*D[L_k]ρ
	with D[L_k]ρ the standard dissipator for the Lindblad operator L_k, we describe each Lindblad operator in the X_i, P_i basis as
	L_k = R_k^⊺*c
	Then
	Z = ∑_k γ_k*conj(R_k)*R_k^⊺

	t0: float, start time

	tf: float, end time
KEYWORD OPTIONAL
	tstep: float, time steps at which output data should be saved
	tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
	alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
"""
function CV_solve_time_dependent(C_in::Array{T1,2},a_in::Vector{Float64},ω::Function,g::Function,λ::Function,α::Function,Z::Array{T2,2},t0::AbstractFloat,tf::AbstractFloat; tstep::AbstractFloat=0.,tols::Vector{Float64}=[1e-6,1e-3],alg = Tsit5()) where {T1 <: Number, T2 <: Number}

	# Conert to X,P basis
	Num = length(ω(0.))
	temp_h1 = zeros(Float64,size(C_in,1),size(C_in,2))
	gs1 = zeros(Float64,Num,Num)
	λs1 = zeros(Float64,Num,Num)

	# function h(t::AbstractFloat;temp_h=temp_h1,gs=gs1,λs=λs1)
	# 	# fill!(temp_h,0.)
	# 	gs[:,:] = g(t)
	# 	λs[:,:] = λ(t)
	#
	# 	for jj = 1:1:Num
	# 		for kk = (jj+1):1:Num
	# 			# XX
	# 			temp_h[2*kk-1,2*jj-1] = (real(gs[kk,jj]) + real(λs[kk,jj]))
	#
	# 			#XY
	# 			temp_h[2*kk-1,2*jj] = (imag(λs[kk,jj]) - imag(gs[kk,jj]))
	#
	# 			#YX
	# 			temp_h[2*kk,2*jj-1] = (imag(λs[kk,jj]) + imag(gs[kk,jj]))
	#
	# 			#YY
	# 			temp_h[2*kk,2*jj] = (real(gs[kk,jj]) - real(λs[kk,jj]))
	# 		end
	#
	# 		#YX same
	# 		temp_h[2*jj,2*jj-1] = 2*imag(λs[jj,jj])
	# 	end
	#
	# 	temp_h[:,:] = temp_h[:,:] + temp_h[:,:]'
	#
	# 	# for ii = 1:1:Num
	# 	# 	temp_h[2*ii-1,2*ii-1] = ω(t)[ii] + 2*real(λ(t)[ii,ii])
	# 	# 	temp_h[2*ii,2*ii] = ω(t)[ii] - 2*real(λ(t)[ii,ii])
	# 	# end
	# 	temp_h[diagind(temp_h)] = vec([(ω(t)+2*real(diag(λs))) (ω(t)-2*real(diag(λs)))]')[:]
	#
	# 	return temp_h
	# end

	function h(t::AbstractFloat)
		# fill!(temp_h,0.)
		gs = g(t)
		λs = λ(t)
		temp_h = zeros(size(gs,1)*2,size(gs,2)*2)

		for jj = 1:1:Num
			for kk = (jj+1):1:Num
				# XX
				temp_h[2*kk-1,2*jj-1] = (real(gs[kk,jj]) + real(λs[kk,jj]))

				#XY
				temp_h[2*kk-1,2*jj] = (imag(λs[kk,jj]) - imag(gs[kk,jj]))

				#YX
				temp_h[2*kk,2*jj-1] = (imag(λs[kk,jj]) + imag(gs[kk,jj]))

				#YY
				temp_h[2*kk,2*jj] = (real(gs[kk,jj]) - real(λs[kk,jj]))
			end

			#YX same
			temp_h[2*jj,2*jj-1] = 2*imag(λs[jj,jj])
		end

		temp_h[:,:] = temp_h[:,:] + temp_h[:,:]'

		# for ii = 1:1:Num
		# 	temp_h[2*ii-1,2*ii-1] = ω(t)[ii] + 2*real(λ(t)[ii,ii])
		# 	temp_h[2*ii,2*ii] = ω(t)[ii] - 2*real(λ(t)[ii,ii])
		# end
		temp_h[diagind(temp_h)] = vec([(ω(t)+2*real(diag(λs))) (ω(t)-2*real(diag(λs)))]')[:]

		return temp_h
	end

	function d(t::AbstractFloat)
		return sqrt(2)*vec([real.(α(t)) -imag.(α(t))]')[:]
	end

	return CV_solve_time_dependent(C_in,a_in,h,d,Z,t0,tf; tstep=tstep,tols=tols,alg=alg)
end
