using Test, DifferentialEquations, LinearAlgebra

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
	stop_points = collect(t0:tstep:tf)
	tols = [1e-7,1e-4]
	rates = [1.,1.,1.,1.]

	tvec1, rho_out1 = MESolve.me_solve_time_independent(rho_in, H, gamma, rates, t0, tf, tstep = tstep, tols = tols)
	tvec2, rho_out2 = MESolve.me_solve_time_independent(rho_in, H, gamma, rates, t0, tf, tstep = tstep, tols = tols, alg = DP5())

	# test time dependent solvers
	Ht(t) = exp(-t)*H

	function H_ip(Htemp::AbstractArray,t::AbstractFloat)
		Htemp[:,:] = Ht(t)[:,:]
		nothing
	end
	rt(t) = exp(-t)*rates

	tvec_t1, rho_out_t1 = MESolve.me_solve_H_time_dependent(rho_in, H_ip, gamma, rates, t0, tf, tstep = tstep, tols = tols, stop_points = stop_points)
	tvec_t2, rho_out_t2 = MESolve.me_solve_H_time_dependent(rho_in, H_ip, gamma, rates, t0, tf, tstep = tstep, tols = tols, alg = DP5(), stop_points = stop_points)
	tvec_t3, rho_out_t3 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols, stop_points = stop_points)
	tvec_t4, rho_out_t4 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5(), stop_points = stop_points)
	tvec_t5, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, H_ip, gamma, rt, t0, tf, tstep = tstep, tols = tols, stop_points = stop_points)
	tvec_t6, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, H_ip, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5(), stop_points = stop_points)

	tvecs = hcat(tvec1,tvec2,tvec_t1,tvec_t2,tvec_t3,tvec_t4,tvec_t5,tvec_t6)

	return tvecs

end

function cv_tests()

	a_in = zeros(6)
	C_in = zeros(ComplexF64,6,6)
	for ii = 1:2:6
    	C_in[ii:(ii+1),ii:(ii+1)] = [1. 1im;-1im 1.0]/2
	end

	ωs = [1.0,2.0,1.0]
	ηs = [0.2,0.1,0.2]
	γs = -ηs + [0.05,0.05,0.05]
	λ_sm = 0.1

	h = [ωs[1] 0. 0. 0. 0. 0.; 0. ωs[1] 0. 0. 0. 0.; ηs[1] 0. (ωs[2] + 2*λ_sm) 0. 0. 0.; 0. γs[1] 0. (ωs[2] - 2*λ_sm) 0. 0.; ηs[2] 0. ηs[3] 0. ωs[3] 0.; 0. γs[2] 0. γs[3] 0. ωs[3]]
	h = h + h' -LinearAlgebra.diagm(LinearAlgebra.diag(h))

	αs = [0.1,0.0,0.2]
	βs = [0.,0.1,0.];
	d = vec([αs βs]')[:]

	L1 = [1;1im;0;0;0;0]/sqrt(2)
	L2 = [0;0;1;1im;0;0]/sqrt(2)
	L3 = [0;0;0;0;1;1im]/sqrt(2)
	Z = 0.05*(conj.(L1)*transpose(L1) + conj.(L2)*transpose(L2) + conj.(L3)*transpose(L3))

	αv = (αs-1im*βs)/sqrt(2)
	g = [0. 0. 0.; ηs[1]+γs[1] 0. 0.; ηs[2]+γs[2] ηs[3]+γs[3] 0.]/2
	λ = [0. 0. 0.; ηs[1]-γs[1] 2*λ_sm 0.; ηs[2]-γs[2] ηs[3]-γs[3] 0.]/2

	hf(t) = h + 0.01*t*h
	df(t) = d + 0.01*t*d

	ωsf(t) = ωs + 0.01*t*ωs
	gf(t) = g + 0.01*t*g
	λf(t) = λ + 0.01*t*λ
	αvf(t) = αv + 0.01*t*αv

	tC1, C_out1, ta1, a_out1 = CV_solve_time_independent(C_in,a_in,h,d,Z,0.0,10.0);
	tC2, C_out2, ta2, a_out2 = CV_solve_time_independent(C_in,a_in,ωs,g,λ,αv,Z,0.0,10.0);
	tC3, C_out3, ta3, a_out3 = CV_solve_time_dependent(C_in,a_in,hf,df,Z,0.0,10.0);

	tC4, C_out4, ta4, a_out4 = CV_solve_time_dependent(C_in,a_in,ωsf,gf,λf,αvf,Z,0.0,10.0);
end

test_solver_args()
cv_tests()
