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

cellmap = HDGElasticity.CellMap([0.,0.],[1.,1.])
ALL = HDGElasticity.LLop(basis,quad,cellmap)
@test allapprox(ALL,0.25*ALLtest)

Dhalf = reshape([1.0],1,1)
cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
Ek = [Dhalf,Dhalf]
func(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
dfunc(x) = [-(1.0-x[2])/4.0,-(1.0-x[1])/4.0]'
ALU = HDGElasticity.LUop(func,dfunc,quad,Ek,cellmap,1)
@test allapprox(ALU,[-2/3],1e-15)

ALU = HDGElasticity.LUop(basis,x->gradient(basis,x),quad,Ek,cellmap,4)
ALUtest = [-2/3 -1/2 -1/2 -1/3
            1/6  0.0  0.0 -1/6
            1/6  0.0  0.0 -1/6
            1/3  1/2  1/2  2/3]
@test allapprox(ALU,ALUtest,1e-15)

Dhalf = diagm(ones(3))
ALU = HDGElasticity.LUop(basis,quad,Dhalf,cellmap)
@test size(ALU) == (12,8)

basis2 = TensorProductBasis(2,2)
quad2 = tensor_product_quadrature(2,3)
ALU = HDGElasticity.LUop(basis2,quad2,Dhalf,cellmap)
@test size(ALU) == (27,18)

basis = TensorProductBasis(2,1)
squad = tensor_product_quadrature(1,2)
facequads = repeat([squad],4)
facemaps = HDGElasticity.reference_cell_facemaps(2)
cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
AUU = HDGElasticity.UUop(basis,facequads,facemaps,4.0,cellmap)

rows = [[4/3,1/3,1/3,0.0],[1/3,4/3,0.0,1/3],[1/3,0.0,4/3,1/3],[0.0,1/3,1/3,4/3]]
AUUtest = 4.0*vcat([HDGElasticity.interpolation_matrix(r,2) for r in rows]...)
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
cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
imap = InterpolatingPolynomial(2,ufs.sbasis)
update!(imap,ufs.icoeffs[1])
normals = reshape(1/sqrt(2)*ones(6),2,3)

func(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
UU = HDGElasticity.UUop(func,ufs.fquads[2,1],
    dgmesh.facemaps,ufs.iquad,normals,imap,1.0,cellmap,1,1)
@test UU[1] ≈ 7/6+4.7/16*sqrt(2.)

Dhalf = HDGElasticity.plane_strain_voigt_hooke_matrix(1.,2.,2)
lop = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],
    ufs.fquads[1,1],dgmesh.facemaps,Dhalf,1.0,cellmap)
@test size(lop) == (20,20)
@test rank(lop) == 20

update!(ufs.imap,ufs.icoeffs[1])
lop = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],ufs.fquads[1,1],
    dgmesh.facemaps,ufs.iquad,ufs.imap,ufs.inormals[1],Dhalf,1.0,cellmap)
@test size(lop) == (20,20)
@test rank(lop) == 20
