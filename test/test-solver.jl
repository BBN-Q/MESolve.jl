using Test, DifferentialEquations

function test_solver_args()
	# test time_independant_solver
	rho_in = zeros(4,4)
	rho_in[1,1] = 1.
	H = zeros(4,4)
	H[1,2] = 1.
	H[2,1] = 1.
	H[2,3] = 1.
	H[3,2] = 1.
	gamma = zeros(4,4,4)
	gamma[1,2,1] = 1.
	t0 = 0.0
	tf = 1.0
	tstep = 0.01
	tols = [1e-4,1e-2]
	rates = [1.,1.,1.,1.]

	tvec1, rho_out1 = MESolve.me_solve_time_independent(rho_in, H, gamma, rates, t0, tf, tstep = tstep, tols = tols)
	tvec2, rho_out2 = MESolve.me_solve_time_independent(rho_in, H, gamma, rates, t0, tf, tstep = tstep, tols = tols, alg = DP5())

	# test time dependent solvers
	Ht(t) = exp(1im*t)*H
	rt(t) = exp(-t)*rates

	tvec_t1, rho_out_t1 = MESolve.me_solve_H_time_dependent(rho_in, Ht, gamma, rates, t0, tf, tstep = tstep, tols = tols)
	tvec_t2, rho_out_t2 = MESolve.me_solve_H_time_dependent(rho_in, Ht, gamma, rates, t0, tf, tstep = tstep, tols = tols, alg = DP5())
	tvec_t3, rho_out_t3 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols)
	tvec_t4, rho_out_t4 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5())
	tvec_t5, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, Ht, gamma, rt, t0, tf, tstep = tstep, tols = tols)
	tvec_t6, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, Ht, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5())

	tvecs = hcat(tvec1,tvec2,tvec_t1,tvec_t2,tvec_t3,tvec_t4,tvec_t5,tvec_t6)

	return tvecs

end

test_solver_args()
