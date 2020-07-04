using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
# using Revise
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

function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
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

f(x) = x^2 - 1.0
r = HDGElasticity.roots_without_end(f,-1.0,1.0,0.0)
@test allapprox(r,[-1.0])

r = HDGElasticity.roots_without_end(f,-1.0,1.5,0.0)
@test allapprox(r,[-1.0,1.0])

r = HDGElasticity.roots_without_start(f,-1.0,1.0,0.0)
@test allapprox(r,[1.0])

r = HDGElasticity.roots_without_start(f,-1.5,1.0,0.0)
@test allapprox(r,[-1.0,1.0])

roots,faceids = HDGElasticity.roots_on_edges(funcs,-1.0,1.0,-1.0,1.0)
testroots = [0.5,0.5]
@test allapprox(roots,testroots)

x1,x2 = HDGElasticity.element_face_intersections(poly,cell)
testx1 = [0.5,-1.0]
testx2 = [0.5,1.0]
@test allapprox(x1,testx1)

x0 = [1.0,0.0]
n = [1.0,1.0]
coeffs = reshape(plane_distance_function(coords,n,x0),NF,:)
update!(poly,coeffs[:,1])
funcs = HDGElasticity.restrict_on_faces(poly,cell)
r,fid = HDGElasticity.roots_on_edges(funcs,-1.0,1.0,-1.0,1.0)
testr = [-1.0,+1.0]
testfid = [2,4]
@test allapprox(r,testr)
@test allequal(fid,testfid)

x1,x2 = HDGElasticity.element_face_intersections(poly,cell)
testx1 = [1.0,-1.0]
testx2 = [-1.0,1.0]
@test allapprox(x1,testx1)
@test allapprox(x2,testx2)

x = [1.0,1.0]
HDGElasticity.gradient_descent_to_zero!(poly,x,1e-3,50)
@test allapprox(x,[0.0,0.0])


function circle_distance_function(x::V,xc,r) where {V<:AbstractVector}
    return (x-xc)'*(x-xc) - r^2
end

function circle_distance_function(x::M,xc,r) where {M<:AbstractMatrix}
    dim,nnodes = size(x)
    vals = zeros(nnodes)
    for idx = 1:nnodes
        vals[idx] = circle_distance_function(x[:,idx],xc,r)
    end
    return vals
end

xc = [1.5,0.5]
radius = 1.2
basis = TensorProductBasis(2,2)
poly = InterpolatingPolynomial(1,basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
coeffs = reshape(circle_distance_function(coords,xc,radius),NF,:)
update!(poly,coeffs[:,1])

x = [1.0,0.0]
HDGElasticity.gradient_descent_to_zero!(poly,x,1e-15,50)
testx = [-0.4,0.0]
@test allapprox(testx,x)

basis = TensorProductBasis(2,1)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
update!(poly,coeffs[:,1])
quad1d = tensor_product_quadrature(1,2)
basis1d = TensorProductBasis(1,1)
mass = HDGElasticity.mass_matrix(basis1d,quad1d,2,1.)
coeffs1d = HDGElasticity.fit_zero_levelset(poly,basis1d,quad1d,mass,cell)
poly1d = InterpolatingPolynomial(2,basis1d)
update!(poly1d,coeffs1d)
@test allapprox(poly1d(-1.0),[0.5,-1.0])
@test allapprox(poly1d(1.0),[0.5,1.0])
