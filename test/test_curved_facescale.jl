using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using HDGElasticity

function curve_length(imap,quad)
    l = 0.0
    for (p,w) in quad
        detjac = HDGElasticity.determinant_jacobian(imap,p)
        l += detjac*w
    end
    return l
end

function curve_length(imap,quad,facescale)
    l = 0.0
    for (idx,(p,w)) in enumerate(quad)
        detjac = HDGElasticity.determinant_jacobian(imap,p)
        l += detjac*facescale[idx]*w
    end
    return l
end

function curve_normals(imap,points)
    npts = length(points)
    normals = zeros(2,npts)
    for (idx,x) in enumerate(points)
        t = gradient(imap,x)
        n = [-t[2],t[1]]
        normals[:,idx] .= n/norm(n)
    end
    return normals
end

imap = InterpolatingPolynomial(2,1,2)
icoeffs = [-1.,-1.,0.,0.,1.,-1.]
update!(imap,icoeffs)
quad = tensor_product_quadrature(1,5)
refl = curve_length(imap,quad)
testlength = sqrt(5) + 0.5*asinh(2.0)
@test isapprox(refl,testlength,atol=1e-2)

cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
refnormals = curve_normals(imap,quad.points)
invjac = HDGElasticity.inverse_jacobian(cellmap)
transformed_normals = diagm(invjac)*refnormals
magnitude = mapslices(x->norm(x),transformed_normals,dims=1)
normals = hcat([transformed_normals[:,i]/magnitude[i] for i = 1:size(magnitude)[2]]...)
jac = HDGElasticity.jacobian(cellmap)
reftangents = hcat([gradient(imap,quad.points[:,i]) for i = 1:length(quad)]...)
tangents = diagm(jac)*reftangents

facescale = HDGElasticity.scale_area(cellmap,normals)
transformed_length = curve_length(imap,quad,facescale)
testlength = sqrt(2) + asinh(1)
@test isapprox(transformed_length,testlength,atol=1e-2)
