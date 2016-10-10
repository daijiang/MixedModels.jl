"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

Members:

- `LMM`: a [`LinearMixedModel`](@ref) - used for the random effects only.
- `β`: the fixed-effects vector
- `β₀`: similar to `β`,  Used in the PIRLS algorithm if step-halving is necessary.
- `θ`: covariance parameter vector
- `η`: linear predictor w/o offset (also used as scratch storage)
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is necessary.
- `resp`: a `GLM.GlmResp` object
- `logd`: a copy of the logdet(LMM) with unit weights
"""

type GeneralizedLinearMixedModel{T<:AbstractFloat} <: MixedModel
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    η::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GLM.GlmResp
    logd::T
end

fixef(m::GeneralizedLinearMixedModel) = m.β

"""
    glmm(f::Formula, fr::ModelFrame, d::Distribution[, l::GLM.Link])

Return a `GeneralizedLinearMixedModel` object.

The value is ready to be `fit!` but has not yet been fit.
"""
function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, l::Link; wt=[], offset=[])
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    wts = isempty(wt) ? ones(nrow(fr)) : Array(wt)
        # the weights argument is forced to be non-empty in the lmm as it will be used later
    LMM = lmm(f, fr; weights = wts)
    setθ!(LMM, getθ(LMM))   # force a decomposition
    y, u = copy(model_response(LMM)), ranef(LMM, uscale=true)
    wts = oftype(y, wts)
            # fit a glm to the fixed-effects only
    gl = glm(LMM.wttrms[end - 1], y, d, l; wts = wts)
    β = coef(gl)
    reweight!(LMM, gl.rr.wrkwt)
    res = GeneralizedLinearMixedModel(LMM, β, similar(β), getθ(LMM), similar(y),
        zeros.(u), u, zeros.(u), gl.rr, logdet(LMM))
    LaplaceDeviance!(res)
    res
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, GLM.canonicallink(d))

Base.logdet{T}(m::GeneralizedLinearMixedModel{T}) = logdet(m.LMM)

"""
    LaplaceDeviance(m::GeneralizedLinearMixedModel)

Return the Laplace approximation to the deviance of `m`.

If the distribution `D` does not have a scale parameter the Laplace approximation
is defined as the squared length of the conditional modes, `u`, plus the determinant
of `Λ'Z'ZΛ + 1`, plus the sum of the squared deviance residuals.
"""
LaplaceDeviance(m::GeneralizedLinearMixedModel) =
    sum(m.resp.devresid) + m.logd + mapreduce(sumabs2, +, m.u)

"""
    LaplaceDeviance!(m::GeneralizedLinearMixedModel)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate `LaplaceDeviance`.
"""
function LaplaceDeviance!(m::GeneralizedLinearMixedModel)
    updateη!(m)
    GLM.wrkresp(vec(m.LMM.trms[end]), GLM.updateμ!(m.resp, m.η))
    #    GLM.wrkresp!(vec(m.LMM.trms[end]), m.resp)
    reevaluateAend!(m.LMM)
    LaplaceDeviance(m)
end

function StatsBase.loglikelihood{T}(m::GeneralizedLinearMixedModel{T})
    accum = zero(T)
    r = m.resp
    D = Distribution(r)
    if D <: Binomial
        for (μ, y, n) in zip(r.mu, r.y, r.wts)
            accum += logpdf(D(round(Int, n), μ), round(Int, y * n))
        end
    else
        for (μ, y) in zip(r.mu, r.y)
            accum += logpdf(D(μ), y)
        end
    end
    accum - (mapreduce(sumabs2, + , m.u) + logdet(m)) / 2
end

lowerbd(m::GeneralizedLinearMixedModel) = vcat(fill(-Inf, size(m.β)), lowerbd(m.LMM))

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η, b, lm, r, u = m.η, m.b, m.LMM, m.resp, m.u
    Λ, trms = lm.Λ, lm.trms
    A_mul_B!(η, trms[end - 1], m.β)
    for i in eachindex(b)
        unscaledre!(η, trms[i], A_mul_B!(Λ[i], copy!(b[i], u[i])))
    end
    m
end

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects and, optionally, the conditional estimates of the fixed-effects.
"""
function pirls!{T}(m::GeneralizedLinearMixedModel{T})
    iter, maxiter, obj = 0, 10, T(-Inf)
    β₀, β, u₀, u = m.β₀, m.β, m.u₀, m.u
    reweight!(m.LMM, m.resp.wrkwt)
    fill!(β, 0)
    copy!(β₀, β)
    for i in eachindex(u)
        fill!(u[i], 0)
        fill!(u₀[i], 0)
    end
    obj₀ = LaplaceDeviance!(m)
    @show obj₀
    while iter < maxiter
        iter += 1
        fixef!(β, m.LMM, true)
        ranef!(u, β, m.LMM, true)      # solve for new values of u        @show u
        @show iter, extrema(β), map(extrema, u)
        obj = LaplaceDeviance!(m)      # update GLM vecs and evaluate Laplace approx
        @show obj
        nhalf = 0
        while obj > obj₀
            nhalf += 1
            if nhalf > 10
                if iter < 2
                    throw(ErrorException("number of averaging steps > 10"))
                end
                break
            end
            map!(+, β, β₀)
            β ./= 2
            for i in eachindex(u)
                ui = u[i]
                ui₀ = u₀[i]
                for j in eachindex(ui)
                    ui[j] += ui₀[j]
                    ui[j] /= 2
                end
            end
            obj = LaplaceDeviance!(m)
            @show obj, nhalf
        end
        if isapprox(obj, obj₀; atol = 0.0001)
            break
        end
        copy!(β₀, β)
        broadcast(copy!, u₀, u)
        obj₀ = obj
        reweight!(m.LMM, m.resp.wrkwt)
    end
    obj
end

"""
    setβθ!{T}(m::GeneralizedLinearMixedModel{T}, v::Vector{T})

Set the parameter vector, `:βθ`, of `m` to `v`.

`βθ` is the concatenation of the fixed-effects, `β`, and the covariance parameter, `θ`.
"""
function setβθ!{T}(m::GeneralizedLinearMixedModel{T}, v::Vector{T})
    β, lm, offset, X = m.β, m.LMM, m.offset, m.offset₀, m.X
    lb = length(β)
    copy!(β, view(v, 1 : lb))
    setθ!(m.LMM, copy!(m.θ, view(v, (lb + 1) : length(v))))
    BLAS.gemv!('N', one(T), X, β, one(T), isempty(offset₀) ? fill!(offset, 0) : copy!(offset, offset₀))
    m
end

sdest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = one(T)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, optimizer=:LN_BOBYQA]])

Optimize the objective function for `m`
"""
function StatsBase.fit!(m::GeneralizedLinearMixedModel, verbose::Bool=false, optimizer::Symbol=:LN_BOBYQA)
    β, lm = m.β, m.LMM
    βθ = vcat(β, getθ(lm))
    opt = NLopt.Opt(optimizer, length(βθ))
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, vcat(fill!(similar(β), -Inf), lowerbd(lm)))
    feval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        if length(g) ≠ 0
            error("gradient not defined for this model")
        end
        feval += 1
        setβθ!(m, x) |> pirls!
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            if length(g) ≠ 0
                error("gradient not defined for this model")
            end
            feval += 1
            val = setβθ!(m, x) |> pirls!
            print("f_$feval: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            val
        end
        NLopt.min_objective!(opt, vobj)
    else
        NLopt.min_objective!(opt, obj)
    end
    fmin, xmin, ret = NLopt.optimize(opt, βθ)
    ## very small parameter values often should be set to zero
#    xmin1 = copy(xmin)
#    modified = false
#    for i in eachindex(xmin1)
#        if 0. < abs(xmin1[i]) < 1.e-5
#            modified = true
#            xmin1[i] = 0.
#        end
#    end
#    if modified  # branch not tested
#        m[:θ] = xmin1
#        ff = objective(m)
#        if ff ≤ (fmin + 1.e-5)  # zero components if increase in objective is negligible
#            fmin = ff
#            copy!(xmin,xmin1)
#        else
#            m[:θ] = xmin
#        end
#    end
    m.LMM.opt = OptSummary(βθ,xmin,fmin,feval,optimizer)
    restoreX!(m)
    if verbose
        println(ret)
    end
    m
end

function VarCorr(m::GeneralizedLinearMixedModel)
    Λ, trms = m.LMM.Λ, m.LMM.trms
    VarCorr(Λ, [string(trms[i].fnm) for i in eachindex(Λ)],
        [trms[i].cnms for i in eachindex(Λ)], NaN)
end

function Base.show{T}(io::IO, m::GeneralizedLinearMixedModel{T}) # not tested
    println(io, "Generalized Linear Mixed Model fit by minimizing the Laplace approximation to the deviance")
    println(io, "  ", m.LMM.formula)
    println(io, "  Distribution: ", Distribution(m.resp))
    println(io, "  Link: ", Link(m.resp), "\n")
    println(io, string("  Deviance (Laplace approximation): ", @sprintf("%.4f", LaplaceDeviance(m))), "\n")

    show(io,VarCorr(m))
    gl = grplevels(m.LMM)
    print(io, "\n Number of obs: ", length(m.η), "; levels of grouping factors: ", gl[1])
    for l in gl[2:end]
        print(io, ", ", l)
    end
    println(io)
    println(io, "\nFixed-effects parameters:")
    show(io, coeftable(m))
end

varest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = one(T)
