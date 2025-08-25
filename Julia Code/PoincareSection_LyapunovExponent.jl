using Plots
gr()

using LinearAlgebra
using Printf
using Statistics
using DifferentialEquations
using ForwardDiff
function safe_sqrt(x)
    return sqrt(max(0, x))
end

# Parameters of the system
En = 1.5 # Energy
a = 1.5   # distance between one fixed particle and the origin
b = 1.5  # Intensity of the magnetic field
r_values = LinRange(-3, 3, 100) # Initial values of y_0

mutable struct SimParams
    a::Float64
    b::Float64
    lyap_sum::Float64
    idx_trajectory::Int64
    t_final_integration::Float64
end

#Hammilton Equations
function hamilton_eqs!(du, u, p::SimParams, t)
    x, y, px, py = u
    denom_arg1 = p.a^2 + 2p.a*x + x^2 + y^2
    denom_arg2 = p.a^2 - 2p.a*x + x^2 + y^2
    den1 = (safe_sqrt(denom_arg1))^3
    den2 = (safe_sqrt(denom_arg2))^3

    den1_inv = den1 > 1e-10 ? 1.0 / den1 : 0.0
    den2_inv = den2 > 1e-10 ? 1.0 / den2 : 0.0

    du[1] = px + p.b*y/2
    du[2] = py - p.b*x/2

    du[3] = (p.b*(2py - p.b*x)/2 + (2x + 2p.a)*den1_inv + (2x - 2p.a)*den2_inv) / 2
    du[4] = (p.b*(-2px - p.b*y)/2 + (2y)*den2_inv + (2y)*den1_inv) / 2
end

# Tangent space equation
function tangent_eqs_AD!(duv, uv, p::SimParams, t)
    R = view(uv, 1:4)
    U = view(uv, 5:8)

    f_hamilton = R_vec -> begin
        out_du = similar(R_vec)
        hamilton_eqs!(out_du, R_vec, p, t)
        return out_du
    end

    J = ForwardDiff.jacobian(f_hamilton, R)

    hamilton_eqs!(view(duv, 1:4), R, p, t)

    mul!(view(duv, 5:8), J, U)
end
# RENORMALIZATION
renorm_interval = 50.0

function lyapunov_callback(integrator)
    U = view(integrator.u, 5:8)
    norm_U = norm(U)

    if norm_U > 0
        U ./= norm_U
        integrator.p.lyap_sum += log(norm_U)
    end
    nothing
end

poincare_data_by_trajectory = Vector{Vector{Tuple{Float64, Float64, Float64}}}() # (y, py, lyap_sum en ese instante)

condition_poincare(u, t, integrator) = u[1]

function affect_poincare!(integrator)
    # Check if crossing is in the positive x direction (px > 0)
    if integrator.u[3] > 0
        current_y = integrator.u[2]
        current_py = integrator.u[4]

        idx_trajectory = integrator.p.idx_trajectory
        current_lyap_sum = integrator.p.lyap_sum
        while length(poincare_data_by_trajectory) < idx_trajectory
            push!(poincare_data_by_trajectory, [])
        end
        push!(poincare_data_by_trajectory[idx_trajectory], (current_y, current_py, current_lyap_sum))
    end
    nothing
end
cb_poincare = ContinuousCallback(condition_poincare, affect_poincare!;
                                 affect_neg! = affect_poincare!,
                                 save_positions=(false, false),
                                 interp_points=50)
cb_renorm = PeriodicCallback(lyapunov_callback, renorm_interval;
                             initial_affect=true,
                             save_positions=(false, false))
callbacks_list = CallbackSet(cb_renorm, cb_poincare)

t_span = (0.0, 10000.0)

println("Starting simulation for $(length(r_values)) initial conditions...")

empty!(poincare_data_by_trajectory)

# Define a color gradient for the 2D plot (by CI)
colors_map_2d_ci = Plots.cgrad(:rainbow, length(r_values), categorical=true)

for (idx, y_0) in enumerate(r_values)
    x_0 = 0.0
    px_0 = 0.0

    discriminant = 2*En-(b^2*y_0^2)/4-4/safe_sqrt(a^2+y_0^2)
    if discriminant < 0
        println("Valor de y_0 = $(@sprintf("%.2f", y_0)) ignorado porque genera condiciones iniciales complejas (discriminante = $(@sprintf("%.2f", discriminant))).")
        push!(poincare_data_by_trajectory, [])
        continue
    end

    py_0 = safe_sqrt(discriminant)

    y0_state = [x_0, y_0, px_0, py_0]

    u0_tangent = [1.0, 0.0, 0.0, 0.0]
    u0_tangent_normalized = u0_tangent / norm(u0_tangent)

    uv0 = vcat(y0_state, u0_tangent_normalized)
    p_params_for_this_trajectory = SimParams(a, b, 0.0, idx, t_span[2])
    prob = ODEProblem(tangent_eqs_AD!, uv0, t_span, p_params_for_this_trajectory)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-8, callback=callbacks_list)
end
println("\n--- Resumen de Resultados ---")
total_poincare_points = sum(length.(poincare_data_by_trajectory))
println("Total de cruces de Poincaré capturados: $total_poincare_points")

all_y_lyap = Float64[]
all_py_lyap = Float64[]
all_lambda_lyap = Float64[]
final_lambdas_by_trajectory = fill(NaN, length(r_values))
println("\n--- Análisis de Exponentes de Lyapunov ---")
for (idx, points_data) in enumerate(poincare_data_by_trajectory)
    if !isempty(points_data)
        final_lyap_sum_for_trajectory = points_data[end][3]
        lambda_val = final_lyap_sum_for_trajectory / t_span[2]
        final_lambdas_by_trajectory[idx] = lambda_val

        if lambda_val <1e-3 
            current_y0_for_idx = r_values[idx]
            println("Para y_0 = $(@sprintf("%.4f", current_y0_for_idx)), el exponente de Lyapunov es \$ \\lambda = $(@sprintf("%.6f", lambda_val)) \$ (considerado cuasiperiódico/regular).")
        end

        for (y, py, _) in points_data 
            push!(all_y_lyap, y)
            push!(all_py_lyap, py)
            push!(all_lambda_lyap, lambda_val)
        end
    end
end

valid_lambdas = filter(!isnan, final_lambdas_by_trajectory)
min_lambda = isempty(valid_lambdas) ? 0.0 : minimum(valid_lambdas)
max_lambda = isempty(valid_lambdas) ? 0.0 : maximum(valid_lambdas)

# Generating plots

if total_poincare_points == 0
    println("\nNo hay puntos válidos en la sección de Poincaré para graficar.")
else
    p_poincare_by_ci_2d = Plots.plot(xlabel="\$y\$", ylabel="\$P_y\$",
                                     title="", 
                                     xlabelfontsize=14, ylabelfontsize=14, titlefontsize=16,
                                     size=(1000, 800),
                                     dpi=300,
                                     left_margin=8Plots.mm,
                                     bottom_margin=8Plots.mm,
                                     top_margin=10Plots.mm,
                                     aspect_ratio=:equal,
                                     legend=false
                                    )

    for (idx, points_data) in enumerate(poincare_data_by_trajectory)
        if !isempty(points_data)
            y_coords = [pt[1] for pt in points_data]
            py_coords = [pt[2] for pt in points_data]

            current_color = colors_map_2d_ci[idx]

            Plots.scatter!(p_poincare_by_ci_2d, y_coords, py_coords,
                           color=current_color,
                           seriesalpha=0.7,
                           marker=:pixel,
                           markersize=1,
                           markerstrokewidth=0,
                           label=""
                          )
        end
    end

    Plots.savefig(p_poincare_by_ci_2d, "Poincare_section_E=$(En)_B=$(b)_a=$(a).png")
    println("Poincaré section saved as Poincare_section_E=$(En)_B=$(b)_a=$(a).png")

    if !isempty(all_y_lyap)
        p_poincare_lyapunov_2d = Plots.plot(xlabel="\$y\$", ylabel="\$P_y\$",
                                            title="", # Added a title
                                            xlabelfontsize=14, ylabelfontsize=14, titlefontsize=16,
                                            size=(1200, 1000),
                                            dpi=300,
                                            left_margin=8Plots.mm,
                                            bottom_margin=8Plots.mm,
                                            top_margin=10Plots.mm,
                                            aspect_ratio=:equal,
                                            legend=false,
                                            colorbar=true,
                                            colorbar_title = "\$\\lambda\$",
                                            clims=(0, 0.1),
                                            color = Plots.cgrad(:inferno)
                                            )

        Plots.scatter!(p_poincare_lyapunov_2d, all_y_lyap, all_py_lyap,
                       marker=:pixel,
                       markersize=1,
                       markerstrokewidth=0,
                       marker_z=all_lambda_lyap,
                       seriesalpha=0.7,
                       label=""
                      )

        Plots.savefig(p_poincare_lyapunov_2d, "Poincare_section_E=$(En)_B=$(b)_a=$(a)_Lyapunov.png")
        println("Poincaré section with Lyapunov exponent plot saved as Poincare_section_E=$(En)_B=$(b)_a=$(a)_Lyapunov.png")
    else
        println("There isn't enough data to generate the plot of Lyapunov's Exponent.")
    end
end