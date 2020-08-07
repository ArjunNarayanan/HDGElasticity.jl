using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity

function allequal(u,v)
    return all(u .== v)
end

function allapprox(u,v)
    return all(u .≈ v)
end

function allapprox(u,v,atol)
    @assert length(u) == length(v)
    return all([isapprox(u[i],v[i],atol=atol) for i in 1:length(u)])
end

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)
basis = TensorProductBasis(2,1)
quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
cellsign = HDGElasticity.cell_signatures(coeffs,poly)
quads = HDGElasticity.element_quadratures(cellsign,coeffs,poly,quad1d)

py = quad1d.points

p1,w1 = ImplicitDomainQuadrature.transform(quad1d,0.50,1.0)
testp1 = [p1[1]  p1[2]  p1[1]  p1[2]
          py[1]  py[1]  py[2]  py[2]]
testw1 = kron(quad1d.weights,w1)
@test allapprox(testp1,quads[1,1].points)
@test allapprox(testw1,quads[1,1].weights)

p2,w2 = ImplicitDomainQuadrature.transform(quad1d,-1.,0.5)
testp2 = [p2[1]  p2[2]  p2[1]  p2[2]
          py[1]  py[1]  py[2]  py[2]]
testw2 = kron(quad1d.weights,w2)
@test allapprox(testp2,quads[2,1].points)
@test allapprox(testw2,quads[2,1].weights)

qp3 = tensor_product_quadrature(2,2)
@test allequal(qp3.points,quads[1,2].points)
@test allequal(qp3.weights,quads[1,2].weights)
@test !isassigned(quads,2,2)

cellsign[1] = +1
HDGElasticity.element_quadratures!(quads,cellsign,coeffs,poly,quad1d)
@test allequal(qp3.points,quads[1,1].points)

update!(poly,coeffs[:,1])
facemaps = HDGElasticity.reference_cell_facemaps(2)
facequads = reshape([Vector{QuadratureRule{1}}(undef,4) for i = 1:4],2,2)
visited = reshape([zeros(Bool,4) for i = 1:4],2,2)
HDGElasticity.update_face_quadratures!(facequads[1,1],poly,facemaps,visited[1,1],+1,quad1d)

testp,testw = ImplicitDomainQuadrature.transform(quad1d,0.5,1.0)
@test allapprox(facequads[1,1][1].points,testp)
@test allapprox(facequads[1,1][1].weights,testw)
@test allapprox(facequads[1,1][3].points,testp)
@test allapprox(facequads[1,1][3].weights,testw)
@test allapprox(facequads[1,1][2].points,quad1d.points)
@test allapprox(facequads[1,1][2].weights,quad1d.weights)
@test length(facequads[1,1][4]) == 0
testvisited = reshape([zeros(Bool,4) for i = 1:4],2,2)
fill!(testvisited[1,1],1)
@test all([allequal(visited[i],testvisited[i]) for i = 1:4])

connectivity = HDGElasticity.cell_connectivity(mesh)
HDGElasticity.update_neighbor_face_quadratures!(facequads,visited,1,1,connectivity,4)

testvisited[1,2][4] = 1
@test all([allequal(visited[i],testvisited[i]) for i = 1:4])
@test facequads[1,1][2] == facequads[1,2][4]


HDGElasticity.update_face_quadratures!(facequads[2,1],poly,facemaps,visited[2,1],-1,quad1d)

testvisited[2,1] .= true
@test allequal(testvisited,visited)

p2,w2 = ImplicitDomainQuadrature.transform(quad1d,-1.,0.5)
@test allequal(facequads[2,1][1].points,p2)
@test allequal(facequads[2,1][1].weights,w2)
@test length(facequads[2,1][2]) == 0
@test allequal(facequads[2,1][3].points,p2)
@test allequal(facequads[2,1][3].weights,w2)

@test allapprox(facequads[2,1][4].points,quad1d.points)
@test allapprox(facequads[2,1][4].weights,quad1d.weights)

HDGElasticity.update_neighbor_face_quadratures!(facequads,visited,2,1,connectivity,4)
@test !isassigned(facequads[2,2],1)
@test !isassigned(facequads[2,2],2)
@test !isassigned(facequads[2,2],3)
@test length(facequads[2,2][4]) == 0

# facequads = HDGElasticity.face_quadratures(isactivecell,isactiveface,connectivity,
#     coeffs,poly,quad1d)
#
# @test allequal(facequads[1,1,1].points,p1)
# @test allequal(facequads[1,1,1].weights,w1)
# @test allequal(facequads[3,1,1].points,p1)
# @test allequal(facequads[3,1,1].weights,w1)
# @test allapprox(facequads[2,1,1].points,quad1d.points)
# @test allapprox(facequads[2,1,1].weights,quad1d.weights)
# @test allequal(visited,testvisited)
# @test !isassigned(facequads,4,1,1)
# @test allequal(facequads[1,2,1].points,p2)
# @test allequal(facequads[1,2,1].weights,w2)
# @test !isassigned(facequads,2,2,1)
# @test allequal(facequads[3,2,1].points,p2)
# @test allequal(facequads[3,2,1].weights,w2)
# @test allapprox(facequads[4,2,1].points,quad1d.points)
# @test allapprox(facequads[4,2,1].weights,quad1d.weights)
# @test !isassigned(facequads,2,2,1)
# @test !isassigned(facequads,4,2,2)
#
# basis1d = TensorProductBasis(1,1)
# quad1d = tensor_product_quadrature(1,2)
# icoeffs = HDGElasticity.interface_coefficients(isactivecell,coeffs,poly,basis1d,quad1d)
# testicoeffs = zeros(4,2)
# testicoeffs[:,1] = [0.5,-1.0,0.5,1.0]
# @test allapprox(icoeffs,testicoeffs)
#
# vbasis = TensorProductBasis(2,4)
# sbasis = TensorProductBasis(1,4)
# poly = InterpolatingPolynomial(1,vbasis)
# quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(5)
# iquad = tensor_product_quadrature(1,5)
# vtpq = tensor_product_quadrature(2,5)
# coords = HDGElasticity.nodal_coordinates(mesh,vbasis)
# NF = HDGElasticity.number_of_basis_functions(vbasis)
# xc = 0.75
# coeffs = reshape(distance_function(coords,xc),NF,:)
# isactivecell = HDGElasticity.active_cells(coeffs,poly)
# vquads = HDGElasticity.element_quadratures(isactivecell,coeffs,poly,quad1d)
# fquads = HDGElasticity.face_quadratures(isactivecell,isactiveface,connectivity,coeffs,poly,quad1d)
# icoeffs = HDGElasticity.interface_coefficients(isactivecell,coeffs,poly,sbasis,iquad)
#
# dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
# cellmap = HDGElasticity.AffineMap(dgmesh.domain[1])
# imap = InterpolatingPolynomial(2,sbasis)
# inormals = HDGElasticity.interface_normals(dgmesh.isactivecell,icoeffs,imap,
#     coeffs,poly,iquad.points,cellmap)
# testnormals = zeros(size(inormals))
# testnormals[1,:,1] .= 1.0
# @test allapprox(testnormals,inormals,1e-15)
#
# ufs = HDGElasticity.UniformFunctionSpace(vbasis,sbasis,vtpq,iquad,vquads,
#     fquads,icoeffs,iquad,imap,inormals)
#
# ufs = HDGElasticity.UniformFunctionSpace(dgmesh,4,5,coeffs,poly)
#
#
# mesh = UniformMesh([0.,0.],[4.,1.],[2,1])
# coords = HDGElasticity.nodal_coordinates(mesh,vbasis)
# function plane_distance_function(coords,n,x0)
#     return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
# end
# testn = [1.,1.]/sqrt(2.)
# coeffs = reshape(plane_distance_function(coords,testn,[0.5,0.]),NF,:)
# dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
# ufs = HDGElasticity.UniformFunctionSpace(dgmesh,4,5,coeffs,poly)
# testnormals = zeros(size(ufs.inormals))
# testnormals[:,:,1] .= 1.0/sqrt(2.)
# @test allapprox(testnormals,ufs.inormals)
