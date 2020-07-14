using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity

function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

vbasis = TensorProductBasis(2,1)
mesh = UniformMesh([0.,0.],[1.,1.],[2,1])
coords = HDGElasticity.nodal_coordinates(mesh,vbasis)
NF = HDGElasticity.number_of_basis_functions(vbasis)
coeffs = reshape(plane_distance_function(coords,[1.,0.],[0.4,0.]),NF,:)
poly = InterpolatingPolynomial(1,vbasis)

dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,2,coeffs,poly)

D1 = HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.)
D2 = HDGElasticity.plane_strain_voigt_hooke_matrix_2d(2.,3.)
cellmap = HDGElasticity.AffineMap(dgmesh.domain[1])

localsolver = HDGElasticity.compute_local_solver_on_cells(dgmesh,ufs,D1,D2,
    cellmap,1.)
