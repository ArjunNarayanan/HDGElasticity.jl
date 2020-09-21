using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

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

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

polyorder = 1
numqp = 2
levelset = InterpolatingPolynomial(1,2,polyorder)
mesh = UniformMesh([0.,0.],[1.,1.],[1,1])
interface_location = 0.5
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
interfacehelid = hybrid_element_numbering.interfacehelid
number_of_hybrid_elements = hybrid_element_numbering.number_of_hybrid_elements
dofsperelement = ufs.dofsperelement

bc_data = Dict(:mixed_components => mixed_bc_components,
               :mixed_values => mixed_bc_values,
               :traction => x->traction_bc_data(x,1.0))

system_matrix = HDGElasticity.SystemMatrix()
system_rhs = HDGElasticity.SystemRHS()

# HDGElasticity.assemble_uniform_boundary_faces!(system_matrix,dgmesh,ufs,
#     cellsolvers,facetohelid,boundary_classifier,bc_data)
HDGElasticity.assemble_cut_boundary_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,boundary_classifier,bc_data)
HDGElasticity.assemble_boundary_faces!(system_rhs,dgmesh,ufs,
    facetohelid,boundary_classifier,bc_data)
HDGElasticity.assemble_incoherent_interface!(system_matrix,dgmesh,ufs,
    cellsolvers,1,4,1:4,8,5:8,dofsperelement)

ndofs = dofsperelement*number_of_hybrid_elements

K = dropzeros!(HDGElasticity.sparse(system_matrix,ndofs))
R = HDGElasticity.rhs(system_rhs,ndofs)
sol = K\R


H1 = sol[1:16]
L1 = cellsolvers[1,1].iLLxLH*H1
S1 = -D1*reshape(L1[1:12],3,:)
U1 = reshape(L1[13:20],2,:)

H2 = sol[17:end]
L2 = cellsolvers[2,1].iLLxLH*H2
S2 = -D2*reshape(L2[1:12],3,:)
U2 = reshape(L2[13:20],2,:)

strain_xx(t,l,m) = t/((l+2m) - l^2/(l+2m))
strain_yy(exx,l,m) = -l/(l+2m)*exx

ex1 = strain_xx(1.0,lambda1,mu1)
ey1 = strain_yy(ex1,lambda1,mu1)
ex2 = strain_xx(1.0,lambda2,mu2)
ey2 = strain_yy(ex2,lambda2,mu2)

ux2 = ex2*0.5
uy2 = ey2*1.0

ux1 = ux2+ex1*0.5
uy1 = ey1*1.0

@test allapprox(0.5(U1[:,1]+U1[:,3]),[ux2,0.0],1e-10)
@test allapprox(0.5(U1[:,2]+U1[:,4]),[ux2,uy1],1e-10)
@test allapprox(0.5(U2[:,1]+U2[:,3]),[ux2,0.0],1e-10)
@test allapprox(0.5(U2[:,2]+U2[:,4]),[ux2,uy2],1e-10)

testS = zeros(3,4)
testS[1,:] .= 1.0
@test allapprox(testS,S1,1e-10)
@test allapprox(testS,S2,1e-10)
