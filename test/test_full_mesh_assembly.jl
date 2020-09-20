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

function interior_operator(ls)
    return ls.LH'*ls.iLLxLH
end

function is_interior_cell(cellconnectivity)
    return all([c[1] != 0 for c in cellconnectivity])
end

function interior_cells(connectivity)
    ncells = length(connectivity)
    isinterior = zeros(Bool,ncells)
    for cellid in 1:ncells
        isinterior[cellid] = is_interior_cell(connectivity[cellid])
    end
    return isinterior
end

function assemble_uniform_interior_faces!(system_matrix,cellsolvers,
    cellsign,cellids,facetohelid,dofsperelement)

    vals1 = vec(interior_operator(cellsolvers[1]))
    vals2 = vec(interior_operator(cellsolvers[2]))
    ncells = length(cellsign)
    for cellid in cellids
        s = cellsign[cellid]
        if s == 1
            helids = facetohelid[1,cellid]
            @assert isnothing(findfirst(x->x==0,helids))
            HDGElasticity.assemble!(system_matrix,vals1,helids,helids,dofsperelement)
        elseif s == -1
            helids = facetohelid[2,cellid]
            @assert isnothing(findfirst(x->x==0,helids))
            HDGElasticity.assemble!(system_matrix,vals2,helids,helids,dofsperelement)
        end
    end
end

function assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,phaseid,cellid)

    dofsperelement = ufs.dofsperelement
    fhelid = facetohelid[phaseid,cellid]
    ihelid = interfacehelid[phaseid,cellid]
    colelids = fhelid[findall(x->x!=0,fhelid)]
    push!(colelids,ihelid)

    for (faceid,helid) in enumerate(fhelid)
        if helid != 0
            HDGElasticity.assemble_traction_face!(system_matrix,
                dgmesh,ufs,cellsolvers,phaseid,cellid,faceid,
                helid,colelids,dofsperelement)
        end
    end
end

function assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,isinteriorcell)

    cellsign = dgmesh.cellsign
    cellids = findall([x==0 for x in cellsign] .& isinteriorcell)

    for cellid in cellids
        assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
            cellsolvers,facetohelid,interfacehelid,1,cellid)
        assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
            cellsolvers,facetohelid,interfacehelid,2,cellid)
    end
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
isinteriorcell = interior_cells(dgmesh.connectivity)
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
cellids = findall(isinteriorcell)
assemble_uniform_interior_faces!(
    system_matrix,cellsolvers,dgmesh.cellsign,cellids,
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
isinteriorcell = interior_cells(dgmesh.connectivity)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
system_matrix = HDGElasticity.SystemMatrix()
cellids = findall(isinteriorcell)
assemble_uniform_interior_faces!(
    system_matrix,cellsolvers,dgmesh.cellsign,cellids,
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
isinteriorcell = interior_cells(dgmesh.connectivity)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
interfacehelid = hybrid_element_numbering.interfacehelid
system_matrix = HDGElasticity.SystemMatrix()
fhelid = facetohelid[1,5]
ihelid = interfacehelid[1,5]
colelids = fhelid[findall(x->x!=0,fhelid)]
push!(colelids,ihelid)
assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,1,5)
