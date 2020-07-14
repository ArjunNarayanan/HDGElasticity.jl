using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,tol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

basis = TensorProductBasis(2,1)
quad = tensor_product_quadrature(2,2)
ALL = HDGElasticity.LLop(basis,quad)
rows = -1.0/9.0*[4.   2   2   1
                 2    4   1   2
                 2    1   4   2
                 1    2   2   4]
ALLtest = vcat([HDGElasticity.interpolation_matrix(rows[i,:],3) for i = 1:4]...)
@test allapprox(ALL,ALLtest)

map = HDGElasticity.AffineMap([0.0,0.0],[1.0,1.0])
ALL = HDGElasticity.LLop(basis,quad,map)
@test allapprox(ALL,0.25*ALLtest)

Dhalf = reshape([1.0],1,1)
map = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
Ek = [Dhalf,Dhalf]
ALUtest = [-2/3 -1/2 -1/2 -1/3
            1/6  0.0  0.0 -1/6
            1/6  0.0  0.0 -1/6
            1/3  1/2  1/2  2/3]
ALU = HDGElasticity.LUop(basis,quad,Dhalf,map,Ek)
@test allapprox(ALU,ALUtest,1e-15)

Dhalf = diagm(ones(3))
ALU = HDGElasticity.LUop(basis,quad,Dhalf,map)
@test size(ALU) == (12,8)

basis2 = TensorProductBasis(2,2)
quad2 = tensor_product_quadrature(2,3)
ALU = HDGElasticity.LUop(basis2,quad2,Dhalf,map)
@test size(ALU) == (27,18)

basis = TensorProductBasis(2,1)
facequad = tensor_product_quadrature(1,2)
fq = [facequad,facequad,facequad,facequad]
isactiveface = [true,true,true,true]
cellmap = HDGElasticity.AffineMap([-1.,-1],[1.,1.])
AUU = HDGElasticity.UUop(basis,fq,isactiveface,cellmap,4.0)

rows = [[4/3,1/3,1/3,0.0],[1/3,4/3,0.0,1/3],[1/3,0.0,4/3,1/3],[0.0,1/3,1/3,4/3]]
AUUtest = 4.0*vcat([HDGElasticity.interpolation_matrix(r,2) for r in rows]...)
@test allapprox(AUU,AUUtest)

AUU = HDGElasticity.UUop(basis,facequad,cellmap,4.0)
@test allapprox(AUU,AUUtest)

function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end



mesh = UniformMesh([-1.,-1.],[2.,2.],[1,1])
basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
normal = [1.,1.]/sqrt(2.)
coeffs = reshape(plane_distance_function(coords,normal,[0.,-1.]),4,1)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,3,coeffs,poly)
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
imap = InterpolatingPolynomial(2,ufs.sbasis)
update!(imap,ufs.icoeffs[:,1])
normals = reshape(1/sqrt(2)*ones(6),2,3)
UU = HDGElasticity.UUop(ufs.vbasis,view(ufs.fquads,:,2,1),
    dgmesh.isactiveface[:,2,1],ufs.iquad,normals,imap,cellmap,1.0)
@test UU[1] ≈ 7/6+4.7/16*sqrt(2.)

Dhalf = HDGElasticity.plane_strain_voigt_hooke_matrix(1.,2.,2)
lop = HDGElasticity.LocalOperator(ufs.vbasis,ufs.vquads[2,1],
    view(ufs.fquads,:,2,1),dgmesh.isactiveface[:,2,1],ufs.iquad,normals,
    imap,cellmap,Dhalf,1.0)
@test size(lop.local_operator) == (20,20)
@test rank(lop.local_operator) == 20

coeffs = reshape(ones(4),4,1)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,3,coeffs,poly)
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
lop = HDGElasticity.LocalOperator(ufs.vbasis,ufs.vquads[1,1],
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],Dhalf,cellmap,1.0)
@test size(lop.local_operator) == (20,20)
@test rank(lop.local_operator) == 20
