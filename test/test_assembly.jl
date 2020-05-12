using Test, LinearAlgebra
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

@test_throws AssertionError HDGElasticity.check_lengths([1,2,3],[1,2],[1.0,2.0,3.0])
@test_throws AssertionError HDGElasticity.check_lengths([1,2],[1,2],[2,3,4])
@test HDGElasticity.check_lengths([1,2],[1,2],[1,2])

matrix = HDGElasticity.SystemMatrix()
@test length(matrix.rows) == 0
@test length(matrix.vals) == 0
@test length(matrix.cols) == 0

@test HDGElasticity.get_dofs_per_element(2,3,4) == 20
@test HDGElasticity.get_dofs_per_element(2,3,9) == 45
@test HDGElasticity.get_dofs_per_element(3,6,8) == 72

@test HDGElasticity.element_dof_start(5,2,3,4) == 81
@test HDGElasticity.element_dof_start(6,3,6,8) == 361

@test HDGElasticity.element_dof_stop(4,2,3,4) == 80
@test HDGElasticity.element_dof_stop(6,3,6,8) == 432

@test all(HDGElasticity.element_dofs(3,2,3,4) .== 41:60)

sdofs = HDGElasticity.element_stress_dofs(3,2,3,4)
testsdofs = [41,42,43,46,47,48,51,52,53,56,57,58]
@test all(sdofs .== testsdofs)

rows = [1,2,3]
cols = [4,5,6]
r,c = HDGElasticity.element_dofs_to_operator_dofs(rows,cols)
testrows = [1,2,3,1,2,3,1,2,3]
testcols = [4,4,4,5,5,5,6,6,6]
@test all(r .== testrows)
@test all(c .== testcols)

hs = HDGElasticity.hybrid_dof_start(3,60,2,2)
hstest = 60+2*2*2+1
@test hs == hstest

hs = HDGElasticity.hybrid_dof_stop(3,60,2,2)
hstest = 60+3*2*2
@test hs == hstest

hdofs = HDGElasticity.hybrid_dofs(5,60,2,2)
hdofstest = (60+4*2*2+1):(60+5*2*2)
@test all(hdofs .== hdofstest)

matrix = HDGElasticity.SystemMatrix()
vals = rand(400)
HDGElasticity.assemble_local_operator!(matrix,vals,1,2,3,4)
rowtest = repeat(1:20,20)
coltest = repeat(1:20,inner=(20,))
@test all(rowtest .== matrix.rows)
@test all(coltest .== matrix.cols)
@test all(vals .≈ matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = rand(400)
HDGElasticity.assemble_local_operator!(matrix,vals,[4,7],2,3,4)
r1 = repeat(61:80,20)
r2 = repeat(121:140,20)
c1 = repeat(61:80,inner=(20,))
c2 = repeat(121:140,inner=(20,))
@test all(matrix.rows .== vcat(r1,r2))
@test all(matrix.cols .== vcat(c1,c2))
@test all(matrix.vals .== vcat(vals,vals))

matrix = HDGElasticity.SystemMatrix()
vals = rand(400)
HDGElasticity.assemble_local_operator!(matrix,vals,2:3,2,3,4)
r1 = repeat(21:40,20)
r2 = repeat(41:60,20)
c1 = repeat(21:40,inner=(20,))
c2 = repeat(41:60,inner=(20,))
@test all(matrix.rows .== vcat(r1,r2))
@test all(matrix.cols .== vcat(c1,c2))
@test all(matrix.vals .== vcat(vals,vals))

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(20,4)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(2,2,3,4)
HDGElasticity.assemble_local_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,3,40,2,3,2)
hdofs = (40+6*4+1):(40+7*4)
r = repeat(edofs,4)
c = repeat(hdofs,inner=(20,))
@test all(r .== matrix.rows)
@test all(c .== matrix.cols)
@test all(vals[3] .== matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(80)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(1,2,3,4)
HDGElasticity.assemble_local_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,1:4,40,2,3,2)
rows = repeat(edofs,16)
h1 = (40+4*4+1):(40+5*4)
c1 = repeat(h1,inner=(20,))
h2 = (40+5*4+1):(40+6*4)
c2 = repeat(h2,inner=(20,))
h3 = (40+6*4+1):(40+7*4)
c3 = repeat(h3,inner=(20,))
h4 = (40+1*4+1):(40+2*4)
c4 = repeat(h4,inner=(20,))
cols = vcat(c1,c2,c3,c4)
v = vcat(vals...)
@test all(rows .== matrix.rows)
@test all(cols .== matrix.cols)
@test all(v .≈ matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(80)) for i = 1:4]
HDGElasticity.assemble_local_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,1:2,1:4,40,2,3,4,2)
h1 = (40+0*4+1):(40+1*4)
c1 = repeat(h1,inner=(20,))
h2 = (40+1*4+1):(40+2*4)
c2 = repeat(h2,inner=(20,))
h3 = (40+2*4+1):(40+3*4)
c3 = repeat(h3,inner=(20,))
h4 = (40+3*4+1):(40+4*4)
c4 = repeat(h4,inner=(20,))
h5 = (40+4*4+1):(40+5*4)
c5 = repeat(h5,inner=(20,))
h6 = (40+5*4+1):(40+6*4)
c6 = repeat(h6,inner=(20,))
h7 = (40+6*4+1):(40+7*4)
c7 = repeat(h7,inner=(20,))
h8 = (40+1*4+1):(40+2*4)
c8 = repeat(h8,inner=(20,))
cols = vcat(c1,c2,c3,c4,c5,c6,c7,c8)

edofs1 = HDGElasticity.element_dofs(1,2,3,4)
r1 = repeat(edofs,16)
edofs2 = HDGElasticity.element_dofs(2,2,3,4)
r2 = repeat(edofs2,16)
rows = vcat(r1,r2)
v = vcat(vals...)
v = vcat(v,v)

@test all(matrix.rows .== rows)
@test all(matrix.cols .== cols)
@test all(matrix.vals .≈ v)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(20,4)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(2,2,3,4)
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,3,40,2,3,2)
hdofs = (40+6*4+1):(40+7*4)
r = repeat(hdofs,20)
c = repeat(edofs,inner=(4,))
@test all(r .== matrix.rows)
@test all(c .== matrix.cols)
@test all(vals[3] .== matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(80)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(1,2,3,4)
cols = repeat(repeat(edofs,inner=(4,)),4)
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,1:4,40,2,3,2)
h1 = (40+4*4+1):(40+5*4)
c1 = repeat(h1,20)
h2 = (40+5*4+1):(40+6*4)
c2 = repeat(h2,20)
h3 = (40+6*4+1):(40+7*4)
c3 = repeat(h3,20)
h4 = (40+1*4+1):(40+2*4)
c4 = repeat(h4,20)
rows = vcat(c1,c2,c3,c4)
v = vcat(vals...)
@test all(rows .== matrix.rows)
@test all(cols .== matrix.cols)
@test all(v .≈ matrix.vals)



matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(80)) for i = 1:4]
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,
    face_to_hybrid_element_number,1:2,1:4,40,2,3,4,2)
h1 = (40+0*4+1):(40+1*4)
c1 = repeat(h1,20)
h2 = (40+1*4+1):(40+2*4)
c2 = repeat(h2,20)
h3 = (40+2*4+1):(40+3*4)
c3 = repeat(h3,20)
h4 = (40+3*4+1):(40+4*4)
c4 = repeat(h4,20)
h5 = (40+4*4+1):(40+5*4)
c5 = repeat(h5,20)
h6 = (40+5*4+1):(40+6*4)
c6 = repeat(h6,20)
h7 = (40+6*4+1):(40+7*4)
c7 = repeat(h7,20)
h8 = (40+1*4+1):(40+2*4)
c8 = repeat(h8,20)
rows = vcat(c1,c2,c3,c4,c5,c6,c7,c8)

edofs1 = HDGElasticity.element_dofs(1,2,3,4)
r1 = repeat(repeat(edofs1,inner=(4,)),4)
edofs2 = HDGElasticity.element_dofs(2,2,3,4)
r2 = repeat(repeat(edofs2,inner=(4,)),4)
cols = vcat(r1,r2)
v = vcat(vals...)
v = vcat(v,v)

@test all(matrix.rows .== rows)
@test all(matrix.cols .== cols)
@test all(matrix.vals .≈ v)
