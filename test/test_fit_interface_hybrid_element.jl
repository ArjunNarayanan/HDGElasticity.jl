using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity

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
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
update!(poly,coeffs[:,1])
cell = HDGElasticity.reference_cell(2)
funcs = HDGElasticity.restrict_on_faces(poly,cell)


roots,faceids = HDGElasticity.roots_of_restrictions(funcs,-1.,1.)
@test allequal(roots,[0.5,0.5])
@test allequal(faceids,[1,3])

@test allequal(HDGElasticity.extend_to_face(roots[1],1,cell),[0.5,-1.0])
@test allequal(HDGElasticity.extend_to_face(roots[1],2,cell),[1.0,0.5])
@test allequal(HDGElasticity.extend_to_face(roots[1],3,cell),[0.5,+1.0])
@test allequal(HDGElasticity.extend_to_face(roots[1],4,cell),[-1.0,0.5])

extended_roots = HDGElasticity.extend_face_roots(2,roots,faceids,cell)
test_extended_roots = [0.5  0.5
                       -1.0 1.0]
@test allequal(test_extended_roots,extended_roots)
