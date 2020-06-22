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

@test allequal(HDGElasticity.extend_to_face(0.1,1,cell),[0.1,-1.0])
@test allequal(HDGElasticity.extend_to_face(0.2,2,cell),[1.0,0.2])
@test allequal(HDGElasticity.extend_to_face(0.3,3,cell),[0.3,+1.0])
@test allequal(HDGElasticity.extend_to_face(0.4,4,cell),[-1.0,0.4])

roots = [0.5,0.8]
faceids = [1,3]
extroots = HDGElasticity.extend_face_roots(roots,faceids,cell)
testextroots = [0.5  0.8
               -1.0 1.0]
@test allequal(testextroots,extroots)

roots,faceids = HDGElasticity.roots_on_edges(funcs,-1.0,1.0,-1.0,1.0)
testroots = [0.5,0.5]
@test allapprox(roots,testroots)

extroots = HDGElasticity.element_face_intersections(poly,cell)
testextroots = [0.5  0.5
                -1.0  1.0]
@test allapprox(extroots,testextroots)
