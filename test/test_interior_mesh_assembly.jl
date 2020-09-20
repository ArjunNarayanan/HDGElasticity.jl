using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
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

polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[1.,1.],[3,3])
interface_location = 0.25
levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x->distance_function(x,interface_location),mesh,levelset.basis
)
dgmesh = HDGElasticity.DGMesh(mesh,levelsetcoeffs,levelset)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
lambda1,mu1 = 1.,2.
lambda2,mu2 = 1.5,2.5
stabilization = 0.1
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda1,mu1))
D2 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda2,mu2))
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
system_matrix = HDGElasticity.SystemMatrix()
HDGElasticity.assemble_uniform_interior_faces!(
    system_matrix,cellsolvers,dgmesh.cellsign,dgmesh.isinteriorcell,
    facetohelid,ufs.dofsperelement
)
dofsperelement = ufs.dofsperelement
hids = facetohelid[1,5]
edofs = vcat([HDGElasticity.element_dofs(h,dofsperelement) for h in hids]...)
vals1 = vec(cellsolvers[1].LH'*cellsolvers[1].iLLxLH)
ndofs = length(edofs)
rows = repeat(edofs,ndofs)
cols = repeat(edofs,inner=ndofs)
@test allequal(system_matrix.rows,rows)
@test allequal(system_matrix.cols,cols)
@test allapprox(system_matrix.vals,vals1)

interface_location = 0.75
levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x->distance_function(x,interface_location),mesh,levelset.basis
)
dgmesh = HDGElasticity.DGMesh(mesh,levelsetcoeffs,levelset)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
system_matrix = HDGElasticity.SystemMatrix()
HDGElasticity.assemble_uniform_interior_faces!(
    system_matrix,cellsolvers,dgmesh.cellsign,dgmesh.isinteriorcell,
    facetohelid,ufs.dofsperelement
)
hids = facetohelid[2,5]
edofs = vcat([HDGElasticity.element_dofs(h,dofsperelement) for h in hids]...)
vals2 = vec(cellsolvers[2].LH'*cellsolvers[2].iLLxLH)
ndofs = length(edofs)
rows = repeat(edofs,ndofs)
cols = repeat(edofs,inner=ndofs)
@test allequal(system_matrix.rows,rows)
@test allequal(system_matrix.cols,cols)
@test allapprox(system_matrix.vals,vals2)


interface_location = 0.5
levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x->distance_function(x,interface_location),mesh,levelset.basis
)
dgmesh = HDGElasticity.DGMesh(mesh,levelsetcoeffs,levelset)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
interfacehelid = hybrid_element_numbering.interfacehelid

system_matrix = HDGElasticity.SystemMatrix()
HDGElasticity.assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,1,5)

fhelid = facetohelid[1,5]
ihelid = interfacehelid[1,5]
rowelids = fhelid[findall(x->x!=0,fhelid)]
colelids = vcat(rowelids,[ihelid])

coldofs = vcat([HDGElasticity.element_dofs(h,dofsperelement) for h in colelids]...)

rdofs1 = HDGElasticity.element_dofs(rowelids[1],dofsperelement)
rt1,ct1 = HDGElasticity.operator_dofs(rdofs1,coldofs)
rm1,cm1 = HDGElasticity.operator_dofs(rdofs1,rdofs1)

rdofs2 = HDGElasticity.element_dofs(rowelids[2],dofsperelement)
rt2,ct2 = HDGElasticity.operator_dofs(rdofs2,coldofs)
rm2,cm2 = HDGElasticity.operator_dofs(rdofs2,rdofs2)

rdofs3 = HDGElasticity.element_dofs(rowelids[3],dofsperelement)
rt3,ct3 = HDGElasticity.operator_dofs(rdofs3,coldofs)
rm3,cm3 = HDGElasticity.operator_dofs(rdofs3,rdofs3)

rows = [rt1;rm1;rt2;rm2;rt3;rm3]
cols = [ct1;cm1;ct2;cm2;ct3;cm3]

@test all(system_matrix.rows .== rows)
@test all(system_matrix.cols .== cols)

system_matrix = HDGElasticity.SystemMatrix()
HDGElasticity.assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,cellsolvers,
    facetohelid,interfacehelid)
