using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

function traction_boundary_classifier(x)
    if x[2] ≈ 0.0 || x[1] ≈ 0.0
        return :mixed
    else
        return :traction
    end
end

function displacement_boundary_classifier(x)
    if x[2] ≈ 0.0 || x[1] ≈ 0.0
        return :mixed
    elseif x[1] ≈ 1.0
        return :displacement
    else
        return :traction
    end
end

function mixed_bc_components(x)
    if x[2] ≈ 0.0
        return [0.0,1.0],[1.,0.]
    elseif x[1] ≈ 0.0
        return [1.,0.],[0.,1.]
    end
end

function mixed_bc_values(x)
    return 0.0,0.0
end

function traction_bc_data(x,xR)
    if x[1] ≈ xR
        return [1.0,0.0]
    else
        return [0.0,0.0]
    end
end

function displacement_bc_data(x,u)
    return [u[1],x[2]*u[2]]
end


function allequal(u,v)
    return all(u .== v)
end

function allapprox(u,v)
    return all(u .≈ v)
end

function allapprox(v1, v2, tol)
    flag = length(v1) == length(v2)
    return flag && all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[1.,1.],[1,1])
interface_location = 2.5
levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x->distance_function(x,interface_location),mesh,levelset.basis
)
dgmesh = HDGElasticity.DGMesh(mesh,levelsetcoeffs,levelset)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,polyorder,numqp,
    levelsetcoeffs,levelset)
lambda1,mu1 = 1.,2.
lambda2,mu2 = 1.,2.
stabilization = 0.1
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda1,mu1))
D2 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda2,mu2))

e11 = 1.0/((lambda2+2mu2) - lambda2^2/(lambda2+2mu2))
e22 = -lambda2/(lambda2+2mu2)*e11
u1 = e11*1
u2 = e22*1

cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid

bc_data = Dict(:mixed_components => mixed_bc_components,
               :mixed_values => mixed_bc_values,
               :traction => x->traction_bc_data(x,1.0))


###################################
# Test Traction BC condition
system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,2,1,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_traction_face!(system_rhs,dgmesh,ufs,[1.,0.],
    facetohelid,2,1,2)

dofsperelement = ufs.dofsperelement
ndofs = dofsperelement*4

K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

testH1 = [0.0 u1  u1  u1 0.0 u1 0.  0.
          0.0 0.0 0.0 u2 u2  u2 0.0 u2]
@test allapprox(U,testH1,1e-10)

############################################
# Test direct global assembly of bc rhs
system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_boundary_faces!(system_rhs,dgmesh,ufs,
    facetohelid,traction_boundary_classifier,bc_data)

dofsperelement = ufs.dofsperelement
ndofs = dofsperelement*4

K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

testH1 = [0.0 u1  u1  u1 0.0 u1 0.  0.
          0.0 0.0 0.0 u2 u2  u2 0.0 u2]
@test allapprox(U,testH1,1e-10)

###############################################################################
# Two adjacent elements on boundary

polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[2.,1.],[2,1])
interface_location = -5.
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
system_rhs = HDGElasticity.SystemRHS()

bc_data = Dict(:mixed_components => mixed_bc_components,
               :mixed_values => mixed_bc_values,
               :traction => x->traction_bc_data(x,2.0))

HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,
    ufs,cellsolvers,facetohelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_boundary_faces!(system_rhs,dgmesh,ufs,
    facetohelid,traction_boundary_classifier,bc_data)

dofsperelement = ufs.dofsperelement
ndofs = dofsperelement*hybrid_element_numbering.number_of_hybrid_elements
K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

testH2 = hcat(testH1,testH1)
testH2[1,9:end] .+= u1
testH2 = testH2[1:2,1:end-2]
@test allapprox(U,testH2,1e-10)

# ###############################################################################
# # Two adjacent elements with coherent interface
polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[2.,1.],[2,1])
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
system_rhs = HDGElasticity.SystemRHS()

HDGElasticity.assemble_cut_boundary_faces!(system_matrix,dgmesh,ufs,cellsolvers,
    facetohelid,interfacehelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_coherent_interface!(system_matrix,dgmesh,ufs,cellsolvers,
    hybrid_element_numbering)
HDGElasticity.assemble_boundary_faces!(system_rhs,dgmesh,ufs,facetohelid,
    traction_boundary_classifier,bc_data)


dofsperelement = ufs.dofsperelement
ndofs = dofsperelement*hybrid_element_numbering.number_of_hybrid_elements
K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

testU1 = [0.0 u1  u1  u1 0.0  u1 0.5u1 0.5u1
          0.0 0.  0.  u2  u2  u2  0.    u2]
testU2 = [0.0 u1  0.0 u1  0.0 0.0 0.5u1 0.5u1
          0.0 0.  u2  u2  0.  u2  0.    u2]
testU3 = [u1 2u1 2u1 2u1 u1 2u1
          0. 0.  0.  u2  u2  u2]
testU = hcat(testU1,testU2,testU3)
@test allapprox(testU,U,1e-10)


###############################################################################
# 9 cell grid
polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[1.,1.],[3,3])
interface_location = 1/5
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
system_rhs = HDGElasticity.SystemRHS()

bc_data = Dict(:mixed_components => mixed_bc_components,
               :mixed_values => mixed_bc_values,
               :traction => x->traction_bc_data(x,1.0))

HDGElasticity.assemble_uniform_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid)

HDGElasticity.assemble_cut_boundary_faces!(system_matrix,dgmesh,ufs,cellsolvers,
    facetohelid,interfacehelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,traction_boundary_classifier,bc_data)
HDGElasticity.assemble_coherent_interface!(system_matrix,dgmesh,ufs,cellsolvers,
    hybrid_element_numbering)
HDGElasticity.assemble_boundary_faces!(system_rhs,dgmesh,ufs,facetohelid,
    traction_boundary_classifier,bc_data)

ndofs = dofsperelement*hybrid_element_numbering.number_of_hybrid_elements
K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

hcoords = HDGElasticity.hybrid_element_coordinates(dgmesh,ufs,hybrid_element_numbering)

disp = copy(hcoords)
disp[1,:] .*= u1
disp[2,:] .*= u2
@test allapprox(disp,U,1e-10)
