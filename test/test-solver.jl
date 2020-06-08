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
	Ht(t) = cos(t)*H

	function H_ip(Htemp::AbstractArray,t::AbstractFloat)
		Htemp[:,:] = H(t)[:,:]
		nothing
	end
	rt(t) = exp(-t)*rates

	tvec_t1, rho_out_t1 = MESolve.me_solve_H_time_dependent(rho_in, H_ip, gamma, rates, t0, tf, tstep = tstep, tols = tols)
	tvec_t2, rho_out_t2 = MESolve.me_solve_H_time_dependent(rho_in, H_ip, gamma, rates, t0, tf, tstep = tstep, tols = tols, alg = DP5())
	tvec_t3, rho_out_t3 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols)
	tvec_t4, rho_out_t4 = MESolve.me_solve_L_time_dependent(rho_in, H, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5())
	tvec_t5, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, Ht, gamma, rt, t0, tf, tstep = tstep, tols = tols)
	tvec_t6, rho_out_t6 = MESolve.me_solve_full_time_dependent(rho_in, Ht, gamma, rt, t0, tf, tstep = tstep, tols = tols, alg = DP5())

	tvecs = hcat(tvec1,tvec2,tvec_t1,tvec_t2,tvec_t3,tvec_t4,tvec_t5,tvec_t6)

	return tvecs

end

function cv_tests()
	ωs = [1.0,2.0,1.0]
	ηs = [0.2,0.1,0.2]
	γs = -ηs + [0.05,0.05,0.05]
	λ_sm = 0.1

	h = [ωs[1] 0. 0. 0. 0. 0.; 0. ωs[1] 0. 0. 0. 0.; ηs[1] 0. (ωs[2] + 2*λ_sm) 0. 0. 0.; 0. γs[1] 0. (ωs[2] - 2*λ_sm) 0. 0.; ηs[2] 0. ηs[3] 0. ωs[3] 0.; 0. γs[2] 0. γs[3] 0. ωs[3]]
	h = h + h' -diagm(diag(h))

	αs = [0.1,0.0,0.2]
	βs = [0.,0.1,0.];
	d = vec([αs βs]')[:]

	L1 = [1;1im;0;0;0;0]/sqrt(2)
	L2 = [0;0;1;1im;0;0]/sqrt(2)
	L3 = [0;0;0;0;1;1im]/sqrt(2)
	Z = 0.05*(conj.(L1)*transpose(L1) + conj.(L2)*transpose(L2) + conj.(L3)*transpose(L3))

	tC1, C_out1, ta1, a_out1 = CV_solve_time_independent(C_in,a_in,h,d,Z,0.0,10.0);

	αv = (αs-1im*βs)/sqrt(2)
	g = [0. 0. 0.; ηs[1]+γs[1] 0. 0.; ηs[2]+γs[2] ηs[3]+γs[3] 0.]/2
	λ = [0. 0. 0.; ηs[1]-γs[1] 2*λ_sm 0.; ηs[2]-γs[2] ηs[3]-γs[3] 0.]/2

	tC2, C_out2, ta2, a_out2 = CV_solve_time_independent(C_in,a_in,ωs,g,λ,αv,Z,0.0,10.0);

	hf(t) = h + 0.01*t*h
	df(t) = d + 0.01*t*d

	tC3, C_out3, ta3, a_out3 = CV_solve_time_dependent(C_in,a_in,hf,df,Z,0.0,10.0);

	ωsf(t) = ωs + 0.01*t*ωs
	gf(t) = g + 0.01*t*g
	λf(t) = λ + 0.01*t*λ
	αvf(t) = αv + 0.01*t*αv

	tC4, C_out4, ta4, a_out4 = CV_solve_time_dependent(C_in,a_in,ωsf,gf,λf,αvf,Z,0.0,10.0);
end

test_solver_args()
cv_tests()
