using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

facequads = [[1,2],[],[1,2,3],[]]
solverid = HDGElasticity.face_to_solverid(facequads)
@test allequal(solverid,[1,0,2,0])

vbasis = TensorProductBasis(2,1)
mesh = UniformMesh([0.,0.],[1.,1.],[2,1])
coords = HDGElasticity.nodal_coordinates(mesh,vbasis)
NF = HDGElasticity.number_of_basis_functions(vbasis)
coeffs = reshape(plane_distance_function(coords,[1.,0.],[0.4,0.]),NF,:)
poly = InterpolatingPolynomial(1,vbasis)

dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,4,coeffs,poly)

l1,m1 = (1.,2.)
l2,m2 = (2.,3.)
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(l1,m1))
D2 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(l2,m2))
stabilization = 1.0
cellmap = HDGElasticity.CellMap(dgmesh.domain[1])

solveridx = HDGElasticity.cell_to_solver_index(dgmesh.cellsign)
testsolveridx = [3 1
                 4 0]
@test allequal(solveridx,testsolveridx)

cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
@test length(cellsolvers.localsolvers) == 4

update!(ufs.imap,ufs.icoeffs[1,1])
LL1 = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],ufs.fquads[1,1],
    dgmesh.facemaps,ufs.iquad,ufs.imap,-ufs.inormals[1],D1,stabilization,cellmap)
fLH1 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.fquads[1,1],
    dgmesh.facemaps,ufs.fnormals,D1,stabilization,cellmap)
iLH1 = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,-ufs.inormals[1],D1,stabilization,cellmap)
LH1 = [hcat(fLH1...) iLH1]
@test allapprox(cellsolvers[1,1].LL,LL1)
@test allapprox(cellsolvers[1,1].LH,LH1)
@test allequal(cellsolvers[1,1].facetosolverid,[1,2,3,0,4])

LL2 = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[2,1],ufs.fquads[2,1],
    dgmesh.facemaps,ufs.iquad,ufs.imap,ufs.inormals[1],D2,stabilization,cellmap)
fLH2 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.fquads[2,1],
    dgmesh.facemaps,ufs.fnormals,D2,stabilization,cellmap)
iLH2 = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,ufs.inormals[1],D2,stabilization,cellmap)
LH2 = [hcat(fLH2...) iLH2]
@test allapprox(cellsolvers[2,1].LL,LL2)
@test allapprox(cellsolvers[2,1].LH,LH2)
@test allapprox(cellsolvers[2,1].facetosolverid,[1,0,2,3,4])

uLL1 = HDGElasticity.local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,
    dgmesh.facemaps,D1,stabilization,cellmap)
ufLH1 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
    dgmesh.facemaps,ufs.fnormals,D1,stabilization,cellmap)
uLH1 = hcat(ufLH1...)
@test allapprox(cellsolvers.localsolvers[1].LL,uLL1)
@test allapprox(cellsolvers.localsolvers[1].LH,uLH1)
@test allequal(cellsolvers.localsolvers[1].facetosolverid,[1,2,3,4])

uLL2 = HDGElasticity.local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,
    dgmesh.facemaps,D2,stabilization,cellmap)
ufLH2 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
    dgmesh.facemaps,ufs.fnormals,D2,stabilization,cellmap)
uLH2 = hcat(ufLH2...)
@test allapprox(cellsolvers.localsolvers[2].LL,uLL2)
@test allapprox(cellsolvers.localsolvers[2].LH,uLH2)
@test allequal(cellsolvers.localsolvers[2].facetosolverid,[1,2,3,4])
