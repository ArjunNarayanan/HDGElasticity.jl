using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

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
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,4,coeffs,poly)

l1,m1 = (1.,2.)
l2,m2 = (2.,3.)
D1 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(l1,m1))
D2 = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(l2,m2))
cellmap = HDGElasticity.AffineMap(dgmesh.domain[1])

localsolver = HDGElasticity.local_operator_on_cells(dgmesh,ufs,D1,D2,
    cellmap,1.)

lhc = HDGElasticity.local_hybrid_operator_on_cells(dgmesh,ufs,D1,D2,
    cellmap,1.)
lhci = HDGElasticity.local_hybrid_operator_on_interfaces(dgmesh,ufs,D1,D2,
    cellmap,1.)


function bc_displacement(coords;alpha=0.1,beta=0.1)
    disp = copy(coords)
    disp[1,:] .*= alpha
    disp[2,:] .*= beta
    return disp
end

function compute_rhs(LUH,disp,ndofs)
    rhs = zeros(ndofs)
    for i in 1:4
        rhs .+= LUH[i]*disp[i]
    end
    return rhs
end

H1c = [0.0 0.5
       0.0 0.0]
H2c = [0.5 0.5
       0.0 1.0]
H3c = [0.0 0.5
       1.0 1.0]
H4c = [0.0 0.0
       0.0 1.0]
HIcoords = [0.4 0.4
            0.0 1.0]
Hcoords = [H1c,H2c,H3c,H4c]

Hdisp = vec.(bc_displacement.(Hcoords))
HIdisp = vec(bc_displacement(HIcoords))
rhs = compute_rhs(lhc[:,2,1],Hdisp,20)
rI = lhci[2,1]*HIdisp

sol = localsolver[2,1].lulop\(rhs+rI)
L = -D2*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(size(L))
testL[1:2,:] .= 2*(l2+m2)*0.1
testU = zeros(size(U))
testU[:,2] = [0.0,0.1]
testU[:,3] = [0.05,0.0]
testU[:,4] = [0.05,0.1]
@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)

rhs = compute_rhs(lhc[:,1,1],Hdisp,20)
rI = lhci[1,1]*HIdisp

sol = localsolver[1,1].lulop\(rhs+rI)
L = -D1*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(size(L))
testL[1:2,:] .= 2*(l1+m1)*0.1
@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)


H1c = [0.5 1.0
       0.0 0.0]
H2c = [1.0 1.0
       0.0 1.0]
H3c = [0.5 1.0
       1.0 1.0]
H4c = [0.5 0.5
       0.0 1.0]

Hdisp = vec.(bc_displacement.(Hcoords))
HIdisp = vec(bc_displacement(HIcoords))
rhs = compute_rhs(lhc[:,1,2],Hdisp,20)

sol = localsolver[1,2].lulop\(rhs)
L = -D1*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(size(L))
testL[1:2,:] .= 2*(l1+m1)*0.1
@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)


hhop = HDGElasticity.hybrid_operator_on_cells(dgmesh,ufs,cellmap,1.)