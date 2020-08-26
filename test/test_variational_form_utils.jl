using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

HDGE = HDGElasticity

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,tol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

m = [1.0 2.0
     3.0 4.0]
v = [5.0,6.0,7.0]
result = [5.0  10.0 6.0  12.0  7.0  14.0
          15.0 20.0 18.0 24.0  21.0 28.0]
@test allequal(HDGE.make_row_matrix(v,m),result)


result = [5.0 0.0 6.0 0.0 7.0 0.0
          0.0 5.0 0.0 6.0 0.0 7.0]
@test allequal(HDGE.interpolation_matrix(v,2),result)
result = [5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0 0.0
          0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0
          0.0 0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0]
@test allequal(HDGE.interpolation_matrix(v,3),result)

Ek = HDGE.vec_to_symm_mat_converter(2)
@test length(Ek) == 2
E1 = [1.0 0.0 0.0
      0.0 0.0 1.0]'
E2 = [0.0 0.0 1.0
      0.0 1.0 0.0]'
@test allequal(Ek[1],E1)
@test allequal(Ek[2],E2)

Ek = HDGElasticity.vec_to_symm_mat_converter(3)
@test length(Ek) == 3
E1 = zeros(3,6)
E1[1,1] = 1.0
E1[2,4] = 1.0
E1[3,5] = 1.0
E1 = E1'
@test allequal(Ek[1],E1)
E2 = zeros(3,6)
E2[1,4] = 1.0
E2[2,2] = 1.0
E2[3,6] = 1.0
E2 = E2'
@test allequal(Ek[2],E2)
E3 = zeros(3,6)
E3[1,5] = 1.0
E3[2,6] = 1.0
E3[3,3] = 1.0
E3 = E3'
@test allequal(Ek[3],E3)


@test_throws ArgumentError HDGE.vec_to_symm_mat_converter(4)
@test_throws ArgumentError HDGE.vec_to_symm_mat_converter(1)

@test HDGE.symmetric_tensor_dimension(2) == 3
@test HDGE.symmetric_tensor_dimension(3) == 6

@test_throws ArgumentError HDGE.symmetric_tensor_dimension(1)
@test_throws ArgumentError HDGE.symmetric_tensor_dimension(4)


basis = TensorProductBasis(1,1)
quad = tensor_product_quadrature(1,2)
matrix = HDGE.mass_matrix(basis,quad,1.0,1)
testmatrix = [2/3 1/3
              1/3 2/3]
@test allapprox(matrix,testmatrix)

vals = [1. 1.]
rhs = HDGE.linear_form(vals,basis,quad)
@test allapprox(rhs,ones(2))

basis = TensorProductBasis(2,1)
facequad = tensor_product_quadrature(1,2)

imap = InterpolatingPolynomial(2,1,1)
coeffs = [1.  1.
          -1. 1.]
update!(imap,coeffs)
N(x) = [(x[1]+1.)*(x[2]+1.)/4.]
matrix = zeros(1,1)
HDGElasticity.update_mass_matrix!(matrix,N,facequad,imap,[1.,1.],1)
@test allapprox(matrix,[2/3])

N(x) = [(x[1]+1.)*(x[2]+1.)/4.]
matrix = zeros(1,1)
linemap = HDGElasticity.LineMap([+1.,-1.],[1.,1.])
HDGElasticity.update_mass_matrix!(matrix,N,facequad,linemap,1.,1)
@test allapprox(matrix,[2/3])

fill!(matrix,0.0)
HDGElasticity.update_mass_matrix!(matrix,N,facequad,linemap,2.,1)
@test allapprox(matrix,[4/3])

matrix = zeros(1,1)
N(x) = [(1.0 - x[1])*(1.0 - x[2])/4.0]
map1 = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
map2 = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
facequads = [facequad,facequad]
facemaps = [map1,map2]
HDGElasticity.update_mass_matrix_on_faces!(matrix,N,facequads,facemaps,[1.,1.],1)
@test allapprox(matrix,[4/3])

fill!(matrix,0.0)
HDGElasticity.update_mass_matrix_on_faces!(matrix,N,facequads,facemaps,[1.,2.],1)
@test allapprox(matrix,[2.])

fill!(matrix,0.0)
map1 = HDGElasticity.LineMap([1.,-1.],[1.,1.])
map2 = HDGElasticity.LineMap([-1.,1.],[1.,1.])
facequads = [facequad,facequad]
facemaps = [map1,map2]
HDGElasticity.update_mass_matrix_on_faces!(matrix,N,facequads,facemaps,[1.,1.],1)
@test allapprox(matrix,[0.0])

N(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
quad = tensor_product_quadrature(1,3)
imap = InterpolatingPolynomial(2,1,1)
icoeffs = [0.0  -1.0
          -1.0   0.0]
update!(imap,icoeffs)
matrix = zeros(1,1)
HDGElasticity.update_mass_matrix!(matrix,N,quad,imap,[1.,1.,1.],1)
@test matrix[1] ≈ 4.7/16*sqrt(2.)


facequad = tensor_product_quadrature(1,2)
map1 = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
map2 = HDGElasticity.LineMap([1.,-1.],[1.,1.])
map3 = HDGElasticity.LineMap([-1.,1.],[1.,1.])
map4 = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
fq = repeat([facequad],4)
fm = [map1,map2,map3,map4]
scale = ones(4)
matrix = HDGElasticity.mass_matrix_on_boundary(basis,fq,fm,scale,1)
testmatrix = [4/3 1/3 1/3 0.
              1/3 4/3 0. 1/3
              1/3 0. 4/3 1/3
              0.  1/3 1/3 4/3]
@test allapprox(matrix,testmatrix)

quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
N(x) = [(1-x[1])*(1-x[2])/4.0]
fq = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads = [fq,fq]
map1 = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
map2 = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
facemaps = [map1,map2]

matrix = zeros(1,1)
HDGElasticity.update_mass_matrix!(matrix,N,fq,map1,1.0,1)
@test allapprox(matrix,[7/12])

basis = TensorProductBasis(2,1)
fq = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads = [fq,fq]
iquad = tensor_product_quadrature(1,3)
map1 = HDGElasticity.LineMap([-1.,-1.],[1.,-1.])
map2 = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
facemaps = [map1,map2]
imap = InterpolatingPolynomial(2,1,1)
icoeffs = [0.0  -1.0
          -1.0   0.0]
update!(imap,icoeffs)
normals = reshape(1/sqrt(2)*ones(6),2,3)
cellmap = HDGElasticity.CellMap([-1.,-1.],[1.,1.])
iscale = HDGElasticity.scale_area(cellmap,normals)

matrix = zeros(4,4)
matrix = HDGElasticity.mass_matrix_on_boundary(basis,facequads,facemaps,[1.,1.],
    iquad,imap,iscale,1,4)
@test matrix[1] ≈ 14/12 + 4.7/16*sqrt(2)

v = [0.0]
quad = tensor_product_quadrature(2,2)
rhsvals = ones(length(quad))'
N(x) = [0.25*(1-x[1])*(1-x[2])]
cellmap = HDGElasticity.CellMap([0.0,-1.0],[1.0,1.0])
HDGElasticity.linear_form!(v,rhsvals,N,cellmap,quad)
@test allapprox(v,[0.25])
