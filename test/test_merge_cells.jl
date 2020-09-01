using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

levelset = InterpolatingPolynomial(1,2,1)
coeffs = [0.0,0.0,1.0,1.0]
update!(levelset,coeffs)
xc = [0.0,0.0]
k = ImplicitDomainQuadrature.height_direction(levelset,xc)
s = sign(levelset(xc))
