using Test
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

vbasis = TensorProductBasis(2,1)
sbasis = TensorProductBasis(1,1)
squad = tensor_product_quadrature(1,2)

LH = zeros(1,1)
NED = [ones(1,1)]
linemap = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
vfunc(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
sfunc(x) = [(1.0-x[1])/2.0]
HDGElasticity.LHop!(LH,vfunc,sfunc,squad,linemap,NED,1.0,1)
@test allapprox(LH,[2/3])

fill!(LH,0.0)
HDGElasticity.LHop!(LH,vfunc,sfunc,squad,linemap,NED,2.0,1)
@test allapprox(LH,[4/3])

imap = InterpolatingPolynomial(2,sbasis)
update!(imap,[0.,-1.,-1.,0.])
vf(x) = [x[1] + x[2]]
sf(x) = [x]
scale = ones(length(squad))
LH = zeros(1,1)
normals = 1.0/sqrt(2)*ones(2,length(squad))
HDGElasticity.LHop!(LH,vf,sf,squad,imap,normals,NED,scale,1)
@test allapprox(LH,[0.0],1e-15)

facemaps = HDGElasticity.reference_cell_facemaps(2)
LH = zeros(4,2)
testjac = 3.0
HDGElasticity.LHop!(LH,vbasis,sbasis,squad,facemaps[1],NED,testjac,1)
LHtest =  testjac*[2/3  1/3
                   0.0  0.0
                   1/3  2/3
                   0.0  0.0]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.LHop!(LH,vbasis,sbasis,squad,facemaps[2],NED,1.0,1)
LHtest = [0.0   0.0
          0.0   0.0
          2/3   1/3
          1/3   2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.LHop!(LH,vbasis,sbasis,squad,facemaps[3],NED,1.0,1)
LHtest =  [0.0  0.0
           2/3  1/3
           0.0  0.0
           1/3  2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.LHop!(LH,vbasis,sbasis,squad,facemaps[4],NED,1.0,1)
LHtest =  [2/3  1/3
           1/3  2/3
           0.0  0.0
           0.0  0.0]
@test allapprox(LH,LHtest)



normals = repeat([1.,1.]/sqrt(2.),inner=(1,length(squad)))
Dhalf = diagm(ones(3))
cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
imap = InterpolatingPolynomial(2,sbasis)
update!(imap,[0.,-1.,-1.,0.])
LH = HDGElasticity.LHop_on_interface(vbasis,sbasis,squad,imap,normals,Dhalf,cellmap)
@test size(LH) == (12,4)


Dhalf = diagm(ones(3))
cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facequads = [squad,squad,squad,squad]
normals = HDGElasticity.reference_normals()
LH = HDGElasticity.LHop(vbasis,sbasis,facequads,facemaps,normals,Dhalf,cellmap)
@test all([size(L) == (12,4) for L in LH])

UH = zeros(1,1)
vfunc(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
sfunc(x) = [(1.0-x[1])/2.0]
linemap = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
HDGElasticity.UHop!(UH,vfunc,sfunc,squad,linemap,1.0,1)
@test allapprox(UH,[2/3])

UH = zeros(1,1)
linemap = HDGElasticity.LineMap([0.,-1.],[1.,-1.])
HDGElasticity.UHop!(UH,vfunc,sfunc,squad,linemap,1.0,1)
@test allapprox(UH,[1/6])

UH = zeros(4,2)
testjac = 2.0
linemap = HDGElasticity.LineMap([1.0,-1.0],[1.0,1.0])
HDGElasticity.UHop!(UH,vbasis,sbasis,squad,linemap,testjac,1)
UHtest = testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test allapprox(UH,UHtest)

vf(x) = [x[1] + x[2]]
sf(x) = [x]
scale = ones(length(squad))
imap = InterpolatingPolynomial(2,1,1)
update!(imap,[0.,-1.,-1.,0.])
UH = reshape([0.0],1,1)
HDGElasticity.UHop!(UH,vf,sf,squad,imap,scale,1)
@test allapprox(UH,[0.0],1e-15)

UH = HDGElasticity.UHop(vbasis,sbasis,facequads,facemaps,1.0,cellmap)
@test all([size(i) == (8,4) for i in UH])

cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
update!(imap,[0.,-1.,-1.,0.])
normals = reshape(1/sqrt(2)*ones(2*length(squad)),2,:)
UH = HDGElasticity.UHop_on_interface(vbasis,sbasis,squad,imap,normals,1.,cellmap)
@test size(UH) == (8,4)

normals = HDGElasticity.reference_normals()
lhc = HDGElasticity.local_hybrid_operator(vbasis,sbasis,facequads,
    facemaps,normals,Dhalf,1.,cellmap)
@test size(lhc) == (20,16)

update!(imap,[0.,-1.,-1.,0.])
normals = 1/sqrt(2)*ones(2,length(squad))
lhc = HDGElasticity.local_hybrid_operator_on_interface(vbasis,sbasis,squad,
    imap,normals,Dhalf,1.0,cellmap)
@test size(lhc) == (20,4)


function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

mesh = UniformMesh([0.,0.],[2.,1.],[1,1])
poly = InterpolatingPolynomial(1,2,1)
coords = HDGElasticity.nodal_coordinates(mesh,poly.basis)
coeffs = reshape(plane_distance_function(coords,[1.,1.]/sqrt(2.),[0.5,0.]),4,1)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,4,coeffs,poly)
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.))
lhc = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,
    ufs.fquads[2,1],dgmesh.facemaps,ufs.fnormals,Dhalf,1.0,cellmap)
@test size(lhc) == (20,8)
