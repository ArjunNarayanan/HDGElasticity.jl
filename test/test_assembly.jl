using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, tol)
    flag = length(v1) == length(v2)
    return flag && all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

function allequal(v1, v2)
    return all(v1 .== v2)
end

matrix = HDGElasticity.SystemMatrix()
@test length(matrix.rows) == 0
@test length(matrix.vals) == 0
@test length(matrix.cols) == 0

oprows = rand(1:20, 5)
opcols = rand(1:20, 5)
opvals = rand(5)
HDGElasticity.assemble!(matrix, oprows, opcols, opvals)
@test allapprox(matrix.rows, oprows)
@test allapprox(matrix.cols, opcols)
@test allapprox(matrix.vals, opvals)

rhs = HDGElasticity.SystemRHS()
@test length(rhs.rows) == 0
@test length(rhs.vals) == 0

rows = [1, 2, 5]
vals = rand(3)
HDGElasticity.assemble!(rhs, rows, vals)
@test allequal(rhs.rows, rows)
@test allapprox(rhs.vals, vals)

@test HDGElasticity.element_dof_start(5, 4) == 17
@test HDGElasticity.element_dof_start(6, 6) == 31

@test HDGElasticity.element_dof_stop(4, 8) == 32
@test HDGElasticity.element_dof_stop(6, 8) == 48

@test allequal(HDGElasticity.element_dofs(3, 5), 11:15)

rows = [1, 2, 3]
cols = [4, 5, 6]
r, c = HDGElasticity.operator_dofs(rows, cols)
testrows = [1, 2, 3, 1, 2, 3, 1, 2, 3]
testcols = [4, 4, 4, 5, 5, 5, 6, 6, 6]
@test allequal(r, testrows)
@test allequal(c, testcols)

matrix = HDGElasticity.SystemMatrix()
vals = rand(36)
HDGElasticity.assemble!(matrix, vals, 2, 3, 6)
rowtest = repeat(7:12, 6)
coltest = repeat(13:18, inner = 6)
@test allequal(rowtest, matrix.rows)
@test allequal(coltest, matrix.cols)
@test allapprox(vals, matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = rand(108)
HDGElasticity.assemble!(matrix, vals, 3, [2, 3, 5], 6)
rowtest = repeat(13:18, 18)
coltest = vcat((
    repeat(7:12, inner = 6),
    repeat(13:18, inner = 6),
    repeat(25:30, inner = 6),
)...)
@test allequal(matrix.rows, rowtest)
@test allequal(matrix.cols, coltest)
@test allapprox(matrix.vals, vals)

matrix = HDGElasticity.SystemMatrix()
vals = rand(256)
HDGElasticity.assemble!(matrix, vals, [1, 2, 3, 4], [1, 2, 3, 4], 4)
rowtest = repeat(1:16, 16)
coltest = repeat(1:16, inner = 16)
@test allequal(matrix.rows, rowtest)
@test allequal(matrix.cols, coltest)
@test allapprox(matrix.vals, vals)

dofspernode = 2
basis = TensorProductBasis(1, 1)
NF = HDGElasticity.number_of_basis_functions(basis)
dofsperelement = dofspernode * NF
quad = tensor_product_quadrature(1, 2)
HH = HDGElasticity.mass_matrix(basis, quad, 1.0, 2)
rhsvals = repeat(quad.points, inner = (2, 1))
rhs = HDGElasticity.linear_form(rhsvals, basis, quad)
sysmatrix = HDGElasticity.SystemMatrix()
HDGElasticity.assemble_displacement_face!(sysmatrix, basis, quad, 1.0,
    1, dofsperelement)
K = dropzeros!(sparse(
    sysmatrix.rows,
    sysmatrix.cols,
    sysmatrix.vals,
    dofsperelement,
    dofsperelement,
))

sol = K\rhs
sol = reshape(sol,2,:)
testsol = [-1. 1.
           -1. 1.]
@test allapprox(sol,testsol)

vbasis = TensorProductBasis(2,1)
vquad = tensor_product_quadrature(2,2)
sbasis = TensorProductBasis(1,1)
NHF = HDGElasticity.number_of_basis_functions(sbasis)
dofsperelement = NHF*2
squad = tensor_product_quadrature(1,2)
facequads = repeat([squad],4)
facemaps = HDGElasticity.reference_cell_facemaps(2)
normals = HDGElasticity.reference_normals()
lambda = 1.
mu = 2.
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda,mu))
stabilization = 1e-3
cellmap = HDGElasticity.CellMap([1.,1.],[2.,3.])
facescale = HDGElasticity.face_determinant_jacobian(cellmap)

solvercomponents = HDGElasticity.LocalSolverComponents(vbasis,vquad,sbasis,
    facequads,facemaps,normals,D1,stabilization,cellmap)

system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

HDGElasticity.assemble_mixed_face!(system_matrix,vbasis,sbasis,facequads[1],
    facemaps[1],normals[1],[0.,1.],[1.,0.],D1,stabilization,facescale[1],
    solvercomponents.iLLxLH,1,1:4,dofsperelement)

HDGElasticity.assemble_mixed_face!(system_matrix,vbasis,sbasis,facequads[4],
    facemaps[4],normals[4],[1.,0.],[0.,1.],D1,stabilization,facescale[4],
    solvercomponents.iLLxLH,4,1:4,dofsperelement)

HDGElasticity.assemble_traction_face!(system_matrix,sbasis,facequads[2],
    facescale[2],stabilization,solvercomponents.fLH[2]',
    solvercomponents.iLLxLH,2,[1,2,3,4],dofsperelement)

HDGElasticity.assemble_traction_face!(system_matrix,sbasis,facequads[3],
    facescale[3],stabilization,solvercomponents.fLH[3]',
    solvercomponents.iLLxLH,3,[1,2,3,4],dofsperelement)

rhsvals = HDGElasticity.linear_form(-[1.,0.],sbasis,squad)
HDGElasticity.assemble!(system_rhs,rhsvals,2,dofsperelement)

K = HDGElasticity.sparse(system_matrix,16)
rhs = HDGElasticity.rhs(system_rhs,16)

sol = K\rhs
L = solvercomponents.iLLxLH*sol
sigma = -D1*reshape(L[1:12],3,:)
displacement = reshape(L[13:20],2,:)

e11 = ((lambda+2mu) - lambda^2/(lambda+2mu))^-1
e22 = -lambda/(lambda+2mu)*e11
u1 = e11*1.0
u2 = e22*2
s11 = 1.0

testsigma = zeros(3,4)
testsigma[1,:] .= 1.0
testu = zeros(2,4)
testu[:,2] .= [0.,u2]
testu[:,3] .= [u1,0.]
testu[:,4] .= [u1,u2]

@test allapprox(testsigma,sigma,1e-14)
@test allapprox(testu,displacement,1e-12)

# ###############################################################################
# # Test a coherent interface
function distance_function(coords,xc)
    return coords[1,:] .- xc
end

polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
NF = HDGElasticity.number_of_basis_functions(levelset.basis)
mesh = UniformMesh([0.,0.],[2.,1.],[1,1])
coords = HDGElasticity.nodal_coordinates(mesh,levelset.basis)
levelsetcoeffs = reshape(distance_function(coords,0.5),NF,1)
dgmesh = HDGElasticity.DGMesh(mesh,levelsetcoeffs,levelset)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
NHF = HDGElasticity.number_of_basis_functions(ufs.sbasis)
dofsperelement = 2*NHF

system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

lambda,mu = 1.,2.
stabilization = 1e-3
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda,mu))
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D1,stabilization)

function assemble_mixed_face!(system_matrix,dgmesh,ufs,phaseid,cellid,
    faceid,Dhalf,stabilization,dcomp,tcomp,cellsolvers,rowelid,
    colelids,dofsperelement)

    vbasis = ufs.vbasis
    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facemap = dgmesh.facemaps[faceid]
    facenormal = ufs.fnormals[faceid]
    facescale = dgmesh.facescale[faceid]
    iLLxLH = cellsolvers[phaseid,cellid].iLLxLH
    HDGElasticity.assemble_mixed_face!(system_matrix,vbasis,sbasis,facequad,facemap,
        facenormal,dcomp,tcomp,Dhalf,stabilization,facescale,iLLxLH,
        rowelid,colelids,dofsperelement)
end

function assemble_traction_face!(system_matrix,dgmesh,ufs,phaseid,cellid,
    faceid,stabilization,cellsolvers,rowelid,colelids,dofsperelement)

    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facescale = dgmesh.facescale[faceid]
    cellsolver = cellsolvers[phaseid,cellid]
    iLLxLH = cellsolver.iLLxLH
    facetosolverid = cellsolver.facetosolverid
    HL = cellsolver.fLH[facetosolverid[faceid]]'
    HDGElasticity.assemble_traction_face!(system_matrix,sbasis,facequad,
        facescale,stabilization,HL,iLLxLH,rowelid,colelids,dofsperelement)
end

function assemble_displacement_face!(system_matrix,dgmesh,ufs,phaseid,cellid,
    faceid,rowelid,dofsperelement)

    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facescale = dgmesh.facescale[faceid]
    HDGElasticity.assemble_displacement_face!(system_matrix,sbasis,facequad,
        facescale,rowelid,dofsperelement)
end

# Cell 1 phase 1
assemble_mixed_face!(system_matrix,dgmesh,ufs,1,1,1,D1,stabilization,
    [0.,1.],[1.,0.],cellsolvers,1,1:4,dofsperelement)
assemble_displacement_face!(system_matrix,dgmesh,ufs,1,1,2,2,dofsperelement)
assemble_traction_face!(system_matrix,dgmesh,ufs,1,1,3,stabilization,
    cellsolvers,3,1:4,dofsperelement)


# Cell 1 phase 2
assemble_mixed_face!(system_matrix,dgmesh,ufs,2,1,1,D1,stabilization,
    [0.,1.],[1.,0.],cellsolvers,5,5:8,dofsperelement)
assemble_traction_face!(system_matrix,dgmesh,ufs,2,1,3,stabilization,
    cellsolvers,6,5:8,dofsperelement)
assemble_mixed_face!(system_matrix,dgmesh,ufs,2,1,4,D1,stabilization,
    [1.,0.],[0.,1.],cellsolvers,7,5:8,dofsperelement)

# Interface condition
update!(ufs.imap,ufs.icoeffs[1])
cellmap = HDGElasticity.CellMap(dgmesh.domain[1])
HH = HDGElasticity.HHop(ufs.sbasis,ufs.iquad,ufs.imap,ufs.inormals[1],cellmap)
HDGElasticity.assemble_coherent_interface!(system_matrix,HH,4,8,dofsperelement)

K = HDGElasticity.sparse(system_matrix,32)
