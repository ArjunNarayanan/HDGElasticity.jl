using Test
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

mesh = UniformMesh([0.,0.],[1.,1.],[2,1])
poly = InterpolatingPolynomial(1,2,1)
coeffs = reshape(ones(8),4,2)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,2,coeffs,poly)

cellmap = HDGElasticity.AffineMap(dgmesh.domain[1])
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.))
stabilization = 1.0

lop = HDGElasticity.LocalOperator(ufs.vbasis,ufs.vquads[1,1],
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],Dhalf,cellmap,1.)
LUop = lop.local_operator
LH = HDGElasticity.LHop_on_active_faces(ufs.vbasis,ufs.sbasis,
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],Dhalf,cellmap)
UH = HDGElasticity.UHop_on_active_faces(ufs.vbasis,ufs.sbasis,
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],cellmap,stabilization)

function bc_displacement(coords;alpha=0.1,beta=0.1)
    disp = copy(coords)
    disp[1,:] .*= alpha
    disp[2,:] .*= beta
    return disp
end

function compute_rhs(LH,UH,disp,ndofs)
    rhs = zeros(ndofs)
    for i in 1:4
        rhs .+= [LH[i];UH[i]]*disp[i]
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
Hcoords = [H1c,H2c,H3c,H4c]
Hdisp = vec.(bc_displacement.(Hcoords))

rhs = compute_rhs(LH,UH,Hdisp,20)
sol = LUop\rhs

L = -Dhalf*reshape(sol[1:12],3,:)
U = reshape(sol[13:20],2,:)

testL = zeros(3,4)
testL[1:2,:] .= 0.6
testU = zeros(2,4)
testU[:,2] .= [0.,0.1]
testU[:,3] .= [0.05,0.0]
testU[:,4] .= [0.05,0.1]

@test allapprox(L,testL,1e-15)
@test allapprox(U,testU,1e-15)


function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

coords = HDGElasticity.nodal_coordinates(mesh,poly.basis)
coeffs = reshape(plane_distance_function(coords,[1.,0.],[0.4,0.]),4,2)
dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
ufs = HDGElasticity.UniformFunctionSpace(dgmesh,1,4,coeffs,poly)

update!(poly,coeffs[:,1])
update!(ufs.imap,ufs.icoeffs[:,1])
mapped_points = hcat([ufs.imap(ufs.iquad.points[:,i]) for i in 1:size(ufs.iquad.points)[2]]...)
normals = -HDGElasticity.levelset_normal(poly,mapped_points,cellmap)
lop = HDGElasticity.LocalOperator(ufs.vbasis,ufs.vquads[1,1],
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],ufs.iquad,normals,
    ufs.imap,Dhalf,cellmap,stabilization)
LUop = lop.local_operator
LH = HDGElasticity.LHop_on_active_faces(ufs.vbasis,ufs.sbasis,
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],Dhalf,cellmap)


LHI = HDGElasticity.LHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,normals,
    Dhalf,ufs.imap,cellmap)
UH = HDGElasticity.UHop_on_active_faces(ufs.vbasis,ufs.sbasis,
    view(ufs.fquads,:,1,1),dgmesh.isactiveface[:,1,1],cellmap,stabilization)
UHI = HDGElasticity.UHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,normals,
    ufs.imap,cellmap,stabilization)
#
# HIrefcoords = reshape(ufs.icoeffs[:,1],2,:)
# HIcoords = hcat([cellmap(HIrefcoords[:,i]) for i in 1:size(HIrefcoords)[2]]...)
# HIdisp = vec(bc_displacement(HIcoords))
#
# rhs = compute_rhs(LH,UH,Hdisp,20)
# rI = [LHI;UHI]*HIdisp
# rhs2 = rhs+rI
#
# sol = LUop\rhs2
# L = -Dhalf*reshape(sol[1:12],3,:)
# U = reshape(sol[13:20],2,:)
#
# testL = zeros(3,4)
# testL[1:2,:] .= 0.6
# testU = zeros(2,4)
# testU[:,2] .= [0.,0.1]
# testU[:,3] .= [0.2,0.]
# testU[:,4] .= [0.2,0.1]
#
# @test allapprox(L,testL,1e-15)
# @test allapprox(U,testU,1e-15)
#
#
# normals = HDGElasticity.levelset_normal(poly,mapped_points,cellmap)
# lop = HDGElasticity.LocalOperator(ufs.vbasis,ufs.vquads[2,1],
#     view(ufs.fquads,:,2,1),dgmesh.isactiveface[:,2,1],ufs.iquad,normals,
#     ufs.imap,Dhalf,cellmap,stabilization)
# LU = lop.local_operator
# LH = HDGElasticity.LHop_on_active_faces(ufs.vbasis,ufs.sbasis,
#     view(ufs.fquads,:,2,1),dgmesh.isactiveface[:,2,1],Dhalf,cellmap)
# LHI = HDGElasticity.LHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,normals,
#     Dhalf,ufs.imap,cellmap)
# UH = HDGElasticity.UHop_on_active_faces(ufs.vbasis,ufs.sbasis,
#     view(ufs.fquads,:,2,1),dgmesh.isactiveface[:,2,1],cellmap,stabilization)
# UHI = HDGElasticity.UHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,normals,
#     ufs.imap,cellmap,stabilization)
#
# rhs = compute_rhs(LH,UH,Hdisp,20)
# rI = [LHI;UHI]*HIdisp
# rhs2 = rhs+rI
#
# sol = LU\rhs2
# L = -Dhalf*reshape(sol[1:12],3,:)
# U = reshape(sol[13:20],2,:)
#
# testL = zeros(3,4)
# testL[1:2,:] .= 0.6
# testU = zeros(2,4)
# testU[:,2] .= [0.,0.1]
# testU[:,3] .= [0.2,0.]
# testU[:,4] .= [0.2,0.1]
#
# @test allapprox(L,testL,1e-12)
# @test allapprox(U,testU,1e-12)
#
#
# Hdisp = vec.(bc_displacement.(Hcoords,beta=0.2))
# HIdisp = vec(bc_displacement(HIcoords,beta=0.2))
#
# rhs = compute_rhs(LH,UH,Hdisp,20)
# rI = [LHI;UHI]*HIdisp
# rhs2 = rhs+rI
#
# sol = LU\rhs2
# L = -Dhalf*reshape(sol[1:12],3,:)
# U = reshape(sol[13:20],2,:)
#
# testL = zeros(3,4)
# testL[1,:] .= 0.7
# testL[2,:] .= 1.1
# testU = zeros(2,4)
# testU[:,2] .= [0.,0.2]
# testU[:,3] .= [0.2,0.]
# testU[:,4] .= [0.2,0.2]
#
# @test allapprox(L,testL,1e-12)
# @test allapprox(U,testU,1e-12)
#
#
# function shear_bc_displacement(coords;alpha=0.1)
#     disp = similar(coords)
#     disp[1,:] = alpha*coords[2,:]
#     disp[2,:] .= 0.0
#     return disp
# end
#
# Hdisp = vec.(shear_bc_displacement.(Hcoords))
# HIdisp = vec(shear_bc_displacement(HIcoords))
#
# rhs = compute_rhs(LH,UH,Hdisp,20)
# rI = [LHI;UHI]*HIdisp
# rhs2 = rhs+rI
#
# sol = LU\rhs2
# L = -Dhalf*reshape(sol[1:12],3,:)
# U = reshape(sol[13:20],2,:)
#
# testL = zeros(3,4)
# testL[3,:] .= 0.2
# testU = zeros(2,4)
# testU[:,2] .= [0.1,0.]
# testU[:,3] .= [0.,0.]
# testU[:,4] .= [0.1,0.]
#
# @test allapprox(L,testL,1e-12)
# @test allapprox(U,testU,1e-12)
