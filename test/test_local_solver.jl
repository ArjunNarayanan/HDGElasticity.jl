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

locops = HDGElasticity.local_operator_on_cells(dgmesh,ufs,
            D1,D2,stabilization)
lochybops = HDGElasticity.local_hybrid_operator_on_cells(dgmesh,ufs,D1,D2,stabilization)

function bc_displacement(coords;alpha=0.1,beta=0.1)
    disp = copy(coords)
    disp[1,:] .*= alpha
    disp[2,:] .*= beta
    return disp
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

isactiveface = [length(fq) > 0 ? true : false for fq in ufs.fquads[2,1]]
faceids = findall(isactiveface)

Hdisp = vcat(vec.(bc_displacement.(Hcoords[faceids]))...)
HIdisp = vec(bc_displacement(HIcoords))
H = vcat(Hdisp,HIdisp)

sidx = solveridx[2,1]
rhs = lochybops[sidx]*H
sol = locops[sidx]\rhs
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

isactiveface = [length(fq) > 0 ? true : false for fq in ufs.fquads[1,1]]
faceids = findall(isactiveface)
Hdisp = vcat(vec.(bc_displacement.(Hcoords[faceids]))...)
H = vcat(Hdisp,HIdisp)

sidx = solveridx[1,1]
rhs = lochybops[sidx]*H
sol = locops[sidx]\rhs
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

Hcoords = [H1c,H2c,H3c,H4c]
H = vcat(vec.(bc_displacement.(Hcoords))...)

sidx = solveridx[1,2]
rhs = lochybops[sidx]*H
sol = locops[sidx]\(rhs)
L = -D1*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(size(L))
testL[1:2,:] .= 2*(l1+m1)*0.1
testU = zeros(2,4)
testU[:,1] = [0.05,0.0]
testU[:,2] = [0.05,0.1]
testU[:,3] = [0.1,0.0]
testU[:,4] = [0.1,0.1]

@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)

locsolver = HDGElasticity.LocalSolver(locops,lochybops,solveridx)
testinvLLxLH = [locops[i]\lochybops[i] for i = 1:length(locops)]
@test length(testinvLLxLH) == length(locsolver.invLLxLH)
@test all([allapprox(locsolver.invLLxLH[i],testinvLLxLH[i]) for i = 1:length(testinvLLxLH)])
