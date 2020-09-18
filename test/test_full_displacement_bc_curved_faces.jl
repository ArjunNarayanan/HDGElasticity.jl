using Test
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using HDGElasticity

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i], v2[i], atol = atol) for i = 1:length(v1)])
end

function circle_distance_function(coords, xc, r)
    dist = [r - norm(coords[:, i] - xc) for i = 1:size(coords)[2]]
end

function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end

function bc_displacement(coords; alpha = 0.1, beta = 0.1)
    disp = copy(coords)
    disp[1, :] .*= alpha
    disp[2, :] .*= beta
    return disp
end

polyorder = 2
numqp = 6
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.0, 2.0))
stabilization = 0.1
normals = HDGElasticity.reference_normals()
levelset = InterpolatingPolynomial(1, 2, polyorder)
mesh = UniformMesh([0.0, 0.0], [1.0, 1.0], [1, 1])
circle_center = [2.5, 0.5]
radius = 2.0
levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x -> circle_distance_function(x, circle_center, radius),
    mesh,
    levelset.basis,
)
# levelsetcoeffs = HDGElasticity.levelset_coefficients(
#     x -> plane_distance_function(x, [1.,0.], [0.5,0.]),
#     mesh,
#     levelset.basis,
# )

dgmesh = HDGElasticity.DGMesh(mesh, levelsetcoeffs, levelset)
ufs = HDGElasticity.UniformFunctionSpace(
    dgmesh,
    polyorder,
    numqp,
    levelsetcoeffs,
    levelset,
)
coords = HDGElasticity.nodal_coordinates(mesh,ufs.vbasis)

function curve_normals(imap,points)
    npts = length(points)
    normals = zeros(2,npts)
    for (idx,x) in enumerate(points)
        t = gradient(imap,x)
        n = [-t[2],t[1]]
        normals[:,idx] .= n/norm(n)
    end
    return normals
end


cellmap = HDGElasticity.CellMap(dgmesh.domain[1])

ufs.icoeffs[1][2] = -1.0
ufs.icoeffs[1][6] = +1.0
update!(ufs.imap, ufs.icoeffs[1])

update!(ufs.imap,ufs.icoeffs[1])
refnormals = curve_normals(ufs.imap,ufs.iquad.points)
invjac = HDGElasticity.inverse_jacobian(cellmap)
transformed_normals = diagm(invjac)*refnormals
magnitude = mapslices(x->norm(x),transformed_normals,dims=1)
inormals = -hcat([transformed_normals[:,i]/magnitude[i] for i = 1:size(magnitude)[2]]...)

LL = HDGElasticity.LLop(ufs.vbasis,ufs.vquads[1,1],cellmap)
LU = HDGElasticity.LUop(ufs.vbasis,ufs.vquads[1,1],Dhalf,cellmap)
fLH = HDGElasticity.LHop(ufs.vbasis,ufs.sbasis,
    ufs.fquads[1,1],dgmesh.facemaps,normals,Dhalf,cellmap)
LH = hcat(fLH...)
iLH = HDGElasticity.LHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,
        ufs.imap,inormals,Dhalf,cellmap)
tLH = hcat(LH,iLH)
UU = HDGElasticity.UUop(ufs.vbasis,ufs.fquads[1,1],dgmesh.facemaps,
    ufs.iquad,inormals,ufs.imap,stabilization,cellmap)
fUH = HDGElasticity.UHop(ufs.vbasis,ufs.sbasis,ufs.fquads[1,1],
    dgmesh.facemaps,stabilization,cellmap)
UH = hcat(fUH...)
iUH = HDGElasticity.UHop_on_interface(ufs.vbasis,ufs.sbasis,ufs.iquad,
    ufs.imap,inormals,stabilization,cellmap)
tUH = hcat(UH,iUH)

HIrefcoords = reshape(ufs.icoeffs[1], 2, :)
HIcoords = hcat([cellmap(HIrefcoords[:, i]) for i = 1:size(HIrefcoords)[2]]...)
HIdisp = vec(bc_displacement(HIcoords))

H1c = [
    0.0 0.5 1.0
    0.0 0.0 0.0
]
H2c = [
    1.0 1.0 1.0
    0.0 0.5 1.0
]
H3c = [
    0.0 0.5 1.0
    1.0 1.0 1.0
]
H4c = [
    0.0 0.0 0.0
    0.0 0.5 1.0
]

Hcoords = [H1c, H2c, H3c, H4c]
isactiveface = [length(fq) > 0 ? true : false for fq in ufs.fquads[1, 1]]
faceids = findall(isactiveface)
Hfacedisp = vcat(vec.(bc_displacement.(Hcoords[faceids]))...)
Hdisp = vcat(Hfacedisp, HIdisp)
Udisp = vec(bc_displacement(coords))

lop = HDGElasticity.local_operator(
    ufs.vbasis,
    ufs.vquads[1, 1],
    ufs.fquads[1, 1],
    dgmesh.facemaps,
    ufs.iquad,
    ufs.imap,
    -inormals,
    Dhalf,
    stabilization,
    cellmap,
)
facelhops = HDGElasticity.local_hybrid_operator(
    ufs.vbasis,
    ufs.sbasis,
    ufs.fquads[1, 1],
    dgmesh.facemaps,
    normals,
    Dhalf,
    stabilization,
    cellmap,
)
lhop = hcat(facelhops...)
ilhop = HDGElasticity.local_hybrid_operator_on_interface(
    ufs.vbasis,
    ufs.sbasis,
    ufs.iquad,
    ufs.imap,
    -inormals,
    Dhalf,
    stabilization,
    cellmap,
)

HIrefcoords = reshape(ufs.icoeffs[1], 2, :)
HIcoords = hcat([cellmap(HIrefcoords[:, i]) for i = 1:size(HIrefcoords)[2]]...)
HIdisp = vec(bc_displacement(HIcoords))

H1c = [
    0.0 0.5 1.0
    0.0 0.0 0.0
]
H2c = [
    1.0 1.0 1.0
    0.0 0.5 1.0
]
H3c = [
    0.0 0.5 1.0
    1.0 1.0 1.0
]
H4c = [
    0.0 0.0 0.0
    0.0 0.5 1.0
]


lochyb = hcat(lhop, ilhop)

rhs = lochyb * Hdisp

sol = lop \ rhs
L = -Dhalf * reshape(sol[1:27], 3, :)
U = reshape(sol[28:45], 2, :)

# testL = zeros(3, 9)
# testL[1:2, :] .= 0.6
# @test allapprox(L, testL, 1e-10)
