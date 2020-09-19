using Test
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using HDGElasticity

function distance_function(coords,xc)
    return coords[1,:] .- xc
end



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
lambda,mu = 1.,2.
stabilization = 0.1
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda,mu))
cellsolvers = HDGElasticity.CellSolvers(dgmesh,ufs,D1,D1,stabilization)

facetohelid = HDGElasticity.cell_hybrid_element_ids(dgmesh.cellsign,
    ufs.isactiveface)
