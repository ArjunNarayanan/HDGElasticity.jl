using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity
import ImplicitDomainQuadrature: extend

function allequal(u,v)
    return all(u .== v)
end

function allapprox(u,v)
    return all(u .≈ v)
end

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)
basis = TensorProductBasis(2,1)
quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
update!(poly,coeffs[:,1])
funcs = HDGElasticity.restrict_on_faces(poly,-1.,1.)

roots,indices = HDGElasticity.roots_of_restrictions(funcs,-1.,1.)
@test roots[1][1] == 0.5
@test roots[3][1] == 0.5
@test allequal(indices,[1,3])

@test allequal(HDGElasticity.extend_to_face(roots[1],1,-1.,1.),[0.5,-1.0])
@test allequal(HDGElasticity.extend_to_face(roots[1],2,-1.,1.),[1.0,0.5])
@test allequal(HDGElasticity.extend_to_face(roots[1],3,-1.,1.),[0.5,+1.0])
@test allequal(HDGElasticity.extend_to_face(roots[1],4,-1.,1.),[-1.0,0.5])

extended_roots = HDGElasticity.extend_face_roots(2,roots,indices,-1.,1.)
