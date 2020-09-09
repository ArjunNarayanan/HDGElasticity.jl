using Test
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

mesh = UniformMesh([0.,0.],[2.,1.],[1,1])
poly = InterpolatingPolynomial(1,2,1)
coeffs = reshape(ones(4),4,1)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,2,coeffs,poly)

cellmap = HDGElasticity.CellMap(dgmesh.domain[1])
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.))
stabilization = 1.0
normals = HDGElasticity.reference_normals()

lop = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],
    ufs.fquads[1,1],dgmesh.facemaps,Dhalf,1.0,cellmap)
lhop = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,
    ufs.fquads[1,1],dgmesh.facemaps,normals,Dhalf,stabilization,cellmap)


function bc_displacement(coords;alpha=0.1,beta=0.1)
    disp = copy(coords)
    disp[1,:] .*= alpha
    disp[2,:] .*= beta
    return disp
end

function compute_rhs(lhop,disp,ndofs)
    H = vcat(disp...)
    return lhop*H
end

H1c = [0. 2.
       0. 0.]
H2c = [2. 2.
       0. 1.]
H3c = [0. 2.
       1. 1.]
H4c = [0. 0.
       0. 1.]
Hcoords = [H1c,H2c,H3c,H4c]
Hdisp = vec.(bc_displacement.(Hcoords))

rhs = compute_rhs(lhop,Hdisp,20)
sol = lop\rhs

L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(3,4)
testL[1:2,:] .= 0.6
testU = zeros(2,4)
testU[:,2] .= [0.,0.1]
testU[:,3] .= [0.2,0.]
testU[:,4] .= [0.2,0.1]

@test allapprox(L,testL,1e-14)
@test allapprox(U,testU,1e-15)


function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

coords = HDGElasticity.nodal_coordinates(mesh,poly.basis)
coeffs = reshape(plane_distance_function(coords,[1.,1.]/sqrt(2.),[0.5,0.]),4,1)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,4,coeffs,poly)

update!(poly,coeffs[:,1])
update!(ufs.imap,ufs.icoeffs[1])
mapped_points = hcat([ufs.imap(ufs.iquad.points[:,i]) for i in 1:size(ufs.iquad.points)[2]]...)
lop = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[1,1],
    ufs.fquads[1,1],dgmesh.facemaps,ufs.iquad,-ufs.inormals[1],
    ufs.imap,Dhalf,stabilization,cellmap)
lhop = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,
    ufs.fquads[1,1],dgmesh.facemaps,normals,Dhalf,
    stabilization,cellmap)
ilhop = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,-ufs.inormals[1],Dhalf,stabilization,cellmap)

HIrefcoords = reshape(ufs.icoeffs[1],2,:)
HIcoords = hcat([cellmap(HIrefcoords[:,i]) for i in 1:size(HIrefcoords)[2]]...)
HIdisp = vec(bc_displacement(HIcoords))

rhs = compute_rhs(lhop,Hdisp,20)
rI = ilhop*HIdisp
rhs2 = rhs+rI

sol = lop\rhs2
L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(3,4)
testL[1:2,:] .= 0.6
testU = zeros(2,4)
testU[:,2] .= [0.,0.1]
testU[:,3] .= [0.2,0.]
testU[:,4] .= [0.2,0.1]

@test allapprox(L,testL,1e-15)
@test allapprox(U,testU,1e-15)


lop = HDGElasticity.local_operator(ufs.vbasis,ufs.vquads[2,1],
    ufs.fquads[2,1],dgmesh.facemaps,ufs.iquad,ufs.inormals[1],
    ufs.imap,Dhalf,stabilization,cellmap)
lhop = HDGElasticity.local_hybrid_operator(ufs.vbasis,ufs.sbasis,
    ufs.fquads[2,1],dgmesh.facemaps,normals,Dhalf,
    stabilization,cellmap)
ilhop = HDGElasticity.local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
    ufs.iquad,ufs.imap,ufs.inormals[1],Dhalf,stabilization,cellmap)

lochyb = hcat(lhop,ilhop)

isactiveface = [length(fq) > 0 ? true : false for fq in ufs.fquads[2,1]]
faceids = findall(isactiveface)
Hfacedisp = vcat(vec.(bc_displacement.(Hcoords[faceids]))...)
Hdisp = vcat(Hfacedisp,HIdisp)

rhs = lochyb*Hdisp

sol = lop\rhs
L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)


Hdisp = vcat(vec.(bc_displacement.(Hcoords[faceids],beta=0.2))...)
HIdisp = vec(bc_displacement(HIcoords,beta=0.2))
H = vcat(Hdisp,HIdisp)

rhs = lochyb*H

sol = lop\rhs
L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(3,4)
testL[1,:] .= 0.7
testL[2,:] .= 1.1
testU = zeros(2,4)
testU[:,2] .= [0.,0.2]
testU[:,3] .= [0.2,0.]
testU[:,4] .= [0.2,0.2]

@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)


function shear_bc_displacement(coords;alpha=0.1)
    disp = similar(coords)
    disp[1,:] = alpha*coords[2,:]
    disp[2,:] .= 0.0
    return disp
end

Hdisp = vcat(vec.(shear_bc_displacement.(Hcoords[faceids]))...)
HIdisp = vec(shear_bc_displacement(HIcoords))

H = vcat(Hdisp,HIdisp)

rhs = lochyb*H

sol = lop\rhs
L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(3,4)
testL[3,:] .= 0.2
testU = zeros(2,4)
testU[:,2] .= [0.1,0.]
testU[:,3] .= [0.,0.]
testU[:,4] .= [0.1,0.]

@test allapprox(L,testL,1e-12)
@test allapprox(U,testU,1e-12)
