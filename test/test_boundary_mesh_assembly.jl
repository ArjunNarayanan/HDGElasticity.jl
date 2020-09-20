using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function boundary_classifier(x)
    if x[2] ≈ 0.0 || x[1] ≈ 0.0
        return :mixed
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

function mixed_bc_data(x)
    return 0.0,0.0
end

function traction_bc_data(x,xR)
    if x[1] ≈ xR
        return [1.0,0.0]
    else
        return [0.0,0.0]
    end
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
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D2,stabilization)
hybrid_element_numbering = HDGElasticity.HybridElementNumbering(dgmesh,ufs)
facetohelid = hybrid_element_numbering.facetohelid
system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

bc_data = Dict(:mixed_components => mixed_bc_components)
HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,2,1,boundary_classifier,bc_data)
HDGElasticity.assemble_traction_face!(system_rhs,dgmesh,ufs,[1.,0.],
    facetohelid,2,1,2)

dofsperelement = ufs.dofsperelement
ndofs = dofsperelement*4
K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R

U = reshape(sol,2,:)

e11 = 1.0/((lambda2+2mu2) - lambda2^2/(lambda2+2mu2))
e22 = -lambda2/(lambda2+2mu2)*e11
u1 = e11*1
u2 = e22*1

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

HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,
    ufs,cellsolvers,facetohelid,boundary_classifier,bc_data)
HDGElasticity.assemble_traction_face!(system_rhs,dgmesh,ufs,[1.,0.],
    facetohelid,1,2,2)

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
