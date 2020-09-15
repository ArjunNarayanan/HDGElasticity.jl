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

solver_components = HDGElasticity.solver_components_on_cells(dgmesh,ufs,
            D1,D2,stabilization)

solver_components = HDGElasticity.solver_components_on_cells(dgmesh,ufs,D1,D2,stabilization)
@test length(solver_components) == 4

update!(ufs.imap,ufs.icoeffs[1,1])
LL1 = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],ufs.fquads[1,1],
    dgmesh.facemaps,ufs.iquad,ufs.imap,-ufs.inormals[1],D1,stabilization,cellmap)
fLH1 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.fquads[1,1],
    dgmesh.facemaps,ufs.fnormals,D1,stabilization,cellmap)
iLH1 = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,-ufs.inormals[1],D1,stabilization,cellmap)
LH1 = [hcat(fLH1...) iLH1]
fHH1 = HDGElasticity.hybrid_operator(ufs.sbasis,ufs.fquads[1,1],1.0,cellmap)
iHH1 = HDGElasticity.hybrid_operator_on_interface(ufs.sbasis,ufs.iquad,
    ufs.imap,-ufs.inormals[1],1.0,cellmap)
HH1 = vcat(fHH1,[iHH1])
@test allapprox(solver_components[solveridx[1,1]].LL,LL1)
@test allapprox(solver_components[solveridx[1,1]].LH,LH1)
@test all([allapprox(solver_components[solveridx[1,1]].fHH[i],HH1[i]) for i = 1:length(HH1)])
@test solver_components[solveridx[1,1]].stabilization == stabilization

LL2 = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[2,1],ufs.fquads[2,1],
    dgmesh.facemaps,ufs.iquad,ufs.imap,ufs.inormals[1],D2,stabilization,cellmap)
fLH2 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.fquads[2,1],
    dgmesh.facemaps,ufs.fnormals,D2,stabilization,cellmap)
iLH2 = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,ufs.inormals[1],D2,stabilization,cellmap)
LH2 = [hcat(fLH2...) iLH2]
fHH2 = HDGElasticity.hybrid_operator(ufs.sbasis,ufs.fquads[2,1],1.0,cellmap)
iHH2 = HDGElasticity.hybrid_operator_on_interface(ufs.sbasis,ufs.iquad,
    ufs.imap,ufs.inormals[1],1.0,cellmap)
HH2 = vcat(fHH2,[iHH2])
@test allapprox(solver_components[solveridx[2,1]].LL,LL2)
@test allapprox(solver_components[solveridx[2,1]].LH,LH2)
@test all([allapprox(solver_components[solveridx[2,1]].fHH[i],HH2[i]) for i = 1:length(HH2)])
@test solver_components[solveridx[2,1]].stabilization == stabilization

uLL1 = HDGElasticity.local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,
    dgmesh.facemaps,D1,stabilization,cellmap)
ufLH1 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
    dgmesh.facemaps,ufs.fnormals,D1,stabilization,cellmap)
uLH1 = hcat(ufLH1...)
ufHH1 = HDGElasticity.hybrid_operator(ufs.sbasis,ufs.ftpq,1.0,cellmap)
@test allapprox(solver_components[1].LL,uLL1)
@test allapprox(solver_components[1].LH,uLH1)
@test all([allapprox(solver_components[1].fHH[i],ufHH1[i]) for i = 1:length(ufHH1)])
@test solver_components[1].stabilization == stabilization

uLL2 = HDGElasticity.local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,
    dgmesh.facemaps,D2,stabilization,cellmap)
ufLH2 = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
    dgmesh.facemaps,ufs.fnormals,D2,stabilization,cellmap)
uLH2 = hcat(ufLH2...)
ufHH2 = HDGElasticity.hybrid_operator(ufs.sbasis,ufs.ftpq,1.0,cellmap)
@test allapprox(solver_components[2].LL,uLL2)
@test allapprox(solver_components[2].LH,uLH2)
@test all([allapprox(solver_components[2].fHH[i],ufHH1[i]) for i = 1:length(ufHH2)])
@test solver_components[2].stabilization == stabilization
