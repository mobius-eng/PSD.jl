__precompile__()
module PSD

import Base: mean

using SymPy
# using DataFrames

# %% General distribution
"""
Abstract type for all distributions.
Type parameter `T` defines the domain: numerical or symbolic
"""
abstract Distribution{T}

abstract UnboundDistribution{T} <: Distribution{T}
abstract BoundDistribution{T} <: Distribution{T}

"""
Probability density function
"""
function pdf(d :: Distribution, x)
	z = symbols("z")
    subs(diff(cdf(d, z), z), z => x)
end

xmin_distribution(d :: Distribution) = 0
xmin_distribution{T <: Real}(d :: Distribution{T}) = zero(T)

xmax_distribution(d :: Distribution{SymPy.Sym}) = oo
xmax_distribution{T <: Real}(d :: Distribution{T}) = convert(T, Inf)

"""
Integrate `f(x) pdf(x)` expression for a given distribution.
"""
function integrate_distribution{T <: Real}(f, d :: Distribution{T}, xmin :: T, xmax :: T)
	quadgk(z -> f(z) * pdf(d,z), xmin, xmax) |> first
end

function integrate_distribution(f, d :: Distribution{SymPy.Sym}, xmin, xmax; assumptions...)
	x = symbols("x"; assumptions...)
	integrate(f(x)*pdf(d,x), (x, xmin, xmax))
end

"""
Mathematical expectation, mean value, of the distribution

Keyword arguements:
- `numerical` (`false`) use numerical integration?
- `xmin = -oo` - minimum value of x
- `xmax = oo` - maximum value of x
- `assumption = false` - add any assumptions about x (e.g., `:positive`)
"""
function mean(d :: Distribution; kv...)
	integrate_distribution(x->x, d, xmin_distribution(d), xmax_distribution(d); kv...)
end

"""
Expectation of ξ². Sometimes it is more efficient to calculate just this
without calculating the full variance (that requires also mean of ξ)
"""
function mean_of_square(d :: Distribution; kv...)
	integrate_distribution(x->x^2, d, xmin_distribution(d), xmax_distribution(d); kv...)
end

"""
Variance of the distribution
"""
function variance(d :: Distribution; kv...)
	x2 = mean_of_square(d; kv...)
    me = mean(d; kv...)
    x2 - me^2
end

function relvar(d :: Distribution; kv...)
	x2 = mean_of_square(d; kv...)
	me = mean(d; kv...)
	x2 / me^2 - (me/me)
end

"""
Returns coefficient of determination R² defined as

R² = 1 - SS(res)/SS(tot)

where

SS(tot) = ∑ (y(i)-E[y])²
SS(res) = ∑(f(i)-y(i))²
"""
function r2{T}(data :: Vector{T}, approximation :: Vector{T})
    data_mean = mean(data)
    ss_tot = sum((y - data_mean)^2 for y ∈ data)
    ss_res = sum((data[i]-approximation[i])^2 for i=1:length(data))
    one(T) - ss_res/ss_tot
end

function r2{T}(y :: Vector{T}, x :: Vector{T}, f :: Function)
    y_mean = mean(y)
    ss_tot = sum((η - y_mean)^2 for η ∈ y)
    ss_res = sum((y[i]-f(x[i]))^2 for i=1:length(y))
    one(T) - ss_res/ss_tot
end


# %% GGS

"""
Gates-Gaudin-Schuhmann distribution

P(x) = (x/xmax)^α
"""
type GGS{T} <: BoundDistribution{T}
    α :: T
    xmax :: T
end

# All properties can be expressed directly:
cdf(d :: GGS, x) = (x / d.xmax)^d.α
pdf(d :: GGS, x) = d.α / d.xmax * (x / d.xmax)^(d.α-1)
mean(d :: GGS) = d.α / (d.α + 1) * d.xmax
variance(d :: GGS) = d.α/((d.α+1)^2 * (d.α+2))*d.xmax^2
relvar(d :: GGS) = 1/(d.α*(d.α+2))

xmax_distribution(d :: GGS) = d.xmax

"""
Fits data representing a CFD to a particular
distribution. Returns a pair (D, R2) where
D is a distribution object and R2 is R² coefficient
of determinition of the fit.
"""
function fit_cdf(::Type{Val{GGS}}, x, y)
    xmax = x[end]
    xw = log.(x / xmax)
    yw = log.(y)

    a, λ = linreg(xw,yw)
    f = z -> λ*z
    (GGS(λ,xmax), r2(yw,xw,f))
end

# %%
# %% Harris
"""
Harris distribution
"""
type Harris{T} <: PSD.BoundDistribution{T}
    n :: T
    s :: T
    xmax :: T
end

xmax_distribution(d :: Harris) = d.xmax

cdf(d :: Harris, x) = 1-(1-(x/d.xmax)^d.s)^d.n

function pdf(d :: Harris, x)
	ξ = x / d.xmax
	ξs = ξ^(d.s-1)
	d.s*d.n/d.xmax*ξs*(1-ξs*ξ)^(d.n-1)
end

mean(d ::Harris) = d.n * d.xmax / (1+d.n*d.s)*beta(1/d.s, d.n)

function variance(d :: Harris)
	me = mean(d)
	(2*d.n*d.xmax^2)/(2+d.n*d.s)*beta(2/d.s,d.n)-me^2
end

function relvar(d :: Harris)
	(2*(1+d.n*d.s)^2)/(d.n*(2+d.n*d.s)) * beta(2/d.s,d.n)/(beta(1/d.s,d.n))^2-1
end

function fit_cdf(::Type{Val{Harris}}, x, y)
    xmax = x[end]
    ξ = x / xmax
    # Stage 1: small ξ ⇒ get s
    ξ_small = filter(u -> u < 0.1, ξ)
    lnξ_small = log.(ξ_small)
    yw_small = [log(log(1.0/(1.0-y[i]))) for i=1:length(ξ_small)]
    (lnn, s) = linreg(lnξ_small, yw_small)
    r2s = r2(yw_small, lnξ_small, z -> s*z + lnn)
    # Stage 2: get n from all the data points
    # except for last where y=1
    arg = log.(1.0 .- ξ[1:end-1] .^ s)
    val = log.(1.0 .- y[1:end-1])
    (_,n) = linreg(arg,val)
    r2n = r2(val, arg, z -> n*z)
    (Harris(n,s,xmax), (r2n,r2s))
end

# %% Log-normal
"""
Log-normal distribution
"""
type LogNormal{T} <: UnboundDistribution{T}
	μ :: T
	σ :: T
end

cdf(d :: LogNormal, x) = 0.5*(1+erf((log(x)-d.μ)/(√2*d.σ)))
pdf(d :: LogNormal, x) = 1/(x * d.σ*√(2π))*exp(-0.5*((log(x)-d.μ)/d.σ)^2)
mean(d :: LogNormal) = exp(d.μ+(d.σ)^2/2)
variance(d :: LogNormal) = (exp((d.σ)^2)-1)*(mean(d))^2
relvar(d :: LogNormal) = exp((d.σ)^2)-1

function fit_cdf(::Type{Val{LogNormal}}, x, y)
    h = erfinv.(2y-1)
    lnx = log.(x)
    (l,k) = linreg(h,lnx)
    σ = k / sqrt(2.0)
    μ = l
    (LogNormal(μ,σ), r2(lnx, h, z -> k*z+l))
end

# %% Rosin-Rammler

"""
Rosin-Rammler distribution
"""
type RosinRammler{T} <: UnboundDistribution{T}
    α :: T
    x632 :: T
end

cdf(d :: RosinRammler, x) = 1 - exp(-(x/d.x632)^d.α)
pdf(d :: RosinRammler, x) = d.α/x * (x/d.x632)^d.α * exp(-(x/d.x632)^d.α)
mean(d :: RosinRammler) = d.x632 * gamma(1+1/d.α)
variance(d :: RosinRammler) = d.x632^2*(gamma(1+2/d.α)-(gamma(1+1/d.α))^2)
relvar(d :: RosinRammler) = gamma(1+2/d.α)/(gamma(1+1/d.α))^2-1

function fit_cdf(::Type{Val{RosinRammler}}, x, y)
    xw = log.(x)
    yw = log.(log.(1.0 ./ (1.0 .- y)))
    (l,k) = linreg(yw, xw)
    α = 1.0/k
    x632 = exp(l)
    (RosinRammler(α,x632), r2(xw, yw, z -> k*z+l))
end

# %% Truncation
"""
Truncate: forward transformation: from 0 < x < ∞ to 0 < y < ymax
"""
truncf(ymax, x) = ymax*x/(ymax+x)
"""
Truncate: backward transformation: from 0 < y < ymax to 0 < x < ∞
"""
truncb(ymax, y) = ymax*y /(ymax - y)

"""
Derivative of truncate backward
"""
dtruncb(ymax, y) = (ymax/(y-ymax))^2

abstract AbstractTruncatedDistribution{T} <: BoundDistribution{T}

type TruncatedDistribution{U,T <: UnboundDistribution{U}} <: AbstractTruncatedDistribution{U}
    d :: T
    xmax :: U
end

xmax_distribution(d :: TruncatedDistribution) = d.xmax

cdf(d::TruncatedDistribution, x) = cdf(d.d, truncb(d.xmax, x))
pdf(d :: TruncatedDistribution, x) = pdf(d.d, truncb(d.xmax, x))*dtruncb(d.xmax, x)

function fit_cdf{U, T <: UnboundDistribution{U}}(::Type{Val{TruncatedDistribution{U,T}}}, x, y)
    xmax = x[end]
    xw = truncb.(xmax, x[1:end-1])
    d, r2d = fit_cdf(Val{T}, xw, y[1:end-1])
    (TruncatedDistribution(d, xmax), r2d)
end

# %% Truncated Rosin-Rammler

const TRR = TruncatedDistribution{Float64, RosinRammler{Float64}}

function TRR(α :: Float64, x632 :: Float64, xmax :: Float64)
	TruncatedDistribution(RosinRammler(α, truncb(xmax, x632)), xmax)
end

cdf(d :: TRR, x) = 1 - exp(-(x*d.d.xmax / (d.d.x632*(d.xmax - x)))^d.d.d.α)

function pdf(td :: TRR, x)
    rr = td.d
    α = rr.α
    x632 = rr.x632
    xmax = td.xmax
    tmp = (x*xmax/(x632*(xmax-x)))^α
    α*xmax*tmp / (x*(xmax-x)) * exp(-tmp)
end

# %% Jumped (discontinuous) distribution
type JumpedDistribution{U, T <: Distribution{U}} <: Distribution{U}
	d :: T{U}
	xmin :: U
end

xmin_distribution(d :: JumpedDistribution) = xmin
xmax_distribution(d :: JumpedDistribution) = xmax_distribution(d.d)


cdf{T}(d :: JumpedDistribution{SymPy.Sym, T}, x) = cdf(d.d,x)*Heaviside(x-d.xmin)

function cdf{U <: Real, T}(d :: JumpedDistribution{U, T}, x)
	if x < d.xmin
		zero(U)
	else
		cdf(d.d, x)
	end
end

function pdf{T}(d :: JumpedDistribution{SymPy.Sym, T}, x)
	pdf(d.d,x)*Heaviside(x-d.xmin)+cdf(d.d,x)*DiracDelta(x-d.xmin)
end

function pdf{U <: Real, T}(d :: JumpedDistribution{U,T}, x)
	if x < d.xmin
		zero(U)
	elseif x == d.xmin
		convert(U, Inf)
	else
		pdf(d.d, x)
	end
end

function integrate_distribution{U <: Real, T}(f, d :: JumpedDistribution{U,T}, xmin, xmax)
	if xmin > d.xmin
		integrate_distribution(f, d.d, xmin, xmax = xmax)
	else
		integrate_distribution(f, d.d, xmin = d.xmin, xmax) + f(d.xmin)*cdf(d.d, d.xmin)
	end
end

# %% Characteristics of the distribution

function particles_number{T <: Real}(φ :: T, d :: Distribution{T})
	(1-φ) * 6 / pi * integrate_distribution(x->x^(-3), d)
end

function particles_number(φ :: SymPy.Sym, d :: Distribution{SymPy.Sym}; assumptions...)
	(1-φ) * 6 / PI * integrate_distribution(x->x^(-3), d; assumptions)
end

function particles_area(φ, d :: Distribution; kv...)
	6(1-φ) * integrate_distribution(x -> x^(-1), d; kv...)
end

# %%

end # module
