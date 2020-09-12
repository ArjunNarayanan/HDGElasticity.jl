using Test
using LinearAlgebra
using CartesianMesh
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allequal(v1,v2)
    return all(v1 .== v2)
end

matrix = HDGElasticity.SystemMatrix()
@test length(matrix.rows) == 0
@test length(matrix.vals) == 0
@test length(matrix.cols) == 0

oprows = rand(1:20,5)
opcols = rand(1:20,5)
opvals = rand(5)
HDGElasticity.update!(matrix,oprows,opcols,opvals)
@test allapprox(matrix.rows,oprows)
@test allapprox(matrix.cols,opcols)
@test allapprox(matrix.vals,opvals)

@test HDGElasticity.element_dof_start(5,4) == 17
@test HDGElasticity.element_dof_start(6,6) == 31

@test HDGElasticity.element_dof_stop(4,8) == 32
@test HDGElasticity.element_dof_stop(6,8) == 48

@test allequal(HDGElasticity.element_dofs(3,5),11:15)

rows = [1,2,3]
cols = [4,5,6]
r,c = HDGElasticity.operator_dofs(rows,cols)
testrows = [1,2,3,1,2,3,1,2,3]
testcols = [4,4,4,5,5,5,6,6,6]
@test allequal(r,testrows)
@test allequal(c,testcols)

matrix = HDGElasticity.SystemMatrix()
vals = rand(36)
HDGElasticity.assemble!(matrix,vals,2,3,6)
rowtest = repeat(7:12,6)
coltest = repeat(13:18,inner=6)
@test allequal(rowtest,matrix.rows)
@test allequal(coltest,matrix.cols)
@test allapprox(vals,matrix.vals)

matrix = HDGElasticity.SystemMatrix()
vals = rand(108)
HDGElasticity.assemble!(matrix,vals,3,[2,3,5],6)
rowtest = repeat(13:18,18)
coltest = vcat((repeat(7:12,inner=6),repeat(13:18,inner=6),repeat(25:30,inner=6))...)
@test allequal(matrix.rows,rowtest)
@test allequal(matrix.cols,coltest)
@test allapprox(matrix.vals,vals)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(20,4)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(2,2,3,4)
HDGElasticity.assemble_local_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,3,40,2,2)
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
    face_to_hybrid_element_number,edofs,2,1:4,40,2,2)
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
    face_to_hybrid_element_number,edofs,2,3,40,2,2)
hdofs = (40+6*4+1):(40+7*4)
r = repeat(hdofs,20)
c = repeat(edofs,inner=(4,))
@test all(r .== matrix.rows)
@test all(c .== matrix.cols)
@test all(vals[3] .== matrix.vals)

x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [1,1]
mesh = UniformMesh(x0,widths,nelements)
matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(20,4)) for i = 1:4]
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,mesh,20,4,2)
@test isempty(matrix.rows)
@test isempty(matrix.cols)
@test isempty(matrix.vals)

x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)
matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(20,4)) for i = 1:4]
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,mesh,40,4,2)
r1 = 45:48
rows = repeat(repeat(r1,20),2)
c1 = repeat(1:20,inner=(4,))
c2 = repeat(21:40,inner=(4,))
cols = [c1;c2]
v = [vals[2];vals[4]]
@test all(matrix.rows .== rows)
@test all(matrix.cols .== cols)
@test all(matrix.vals .≈ v)

matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(80)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
edofs = HDGElasticity.element_dofs(1,2,3,4)
cols = repeat(repeat(edofs,inner=(4,)),4)
HDGElasticity.assemble_hybrid_local_operator!(matrix,vals,
    face_to_hybrid_element_number,edofs,2,1:4,40,2,2)
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


matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(4,4)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
HDGElasticity.assemble_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,2,3,40,2,2)
hdofs = (40+6*4+1):(40+7*4)
r = repeat(hdofs,4)
c = repeat(hdofs,inner=(4,))
@test all(r .== matrix.rows)
@test all(c .== matrix.cols)
@test all(vals[3] .== matrix.vals)



matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(16)) for i = 1:4]
face_to_hybrid_element_number = [1  5
                                 2  6
                                 3  7
                                 4  2]
HDGElasticity.assemble_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,2,1:4,40,2,2)
h1 = (40+4*4+1):(40+5*4)
r1 = repeat(h1,4)
c1 = repeat(h1,inner=(4,))
h2 = (40+5*4+1):(40+6*4)
r2 = repeat(h2,4)
c2 = repeat(h2,inner=(4,))
h3 = (40+6*4+1):(40+7*4)
r3 = repeat(h3,4)
c3 = repeat(h3,inner=(4,))
h4 = (40+1*4+1):(40+2*4)
r4 = repeat(h4,4)
c4 = repeat(h4,inner=(4,))
rows = vcat(r1,r2,r3,r4)
cols = vcat(c1,c2,c3,c4)
v = vcat(vals...)
@test all(rows .== matrix.rows)
@test all(cols .== matrix.cols)
@test all(v .≈ matrix.vals)



matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(16)) for i = 1:4]
HDGElasticity.assemble_hybrid_operator!(matrix,vals,
    face_to_hybrid_element_number,1:2,1:4,40,2,2)
h1 = (40+0*4+1):(40+1*4)
c1 = repeat(h1,inner=(4,))
r1 = repeat(h1,4)
h2 = (40+1*4+1):(40+2*4)
c2 = repeat(h2,inner=(4,))
r2 = repeat(h2,4)
h3 = (40+2*4+1):(40+3*4)
c3 = repeat(h3,inner=(4,))
r3 = repeat(h3,4)
h4 = (40+3*4+1):(40+4*4)
c4 = repeat(h4,inner=(4,))
r4 = repeat(h4,4)
h5 = (40+4*4+1):(40+5*4)
c5 = repeat(h5,inner=(4,))
r5 = repeat(h5,4)
h6 = (40+5*4+1):(40+6*4)
c6 = repeat(h6,inner=(4,))
r6 = repeat(h6,4)
h7 = (40+6*4+1):(40+7*4)
c7 = repeat(h7,inner=(4,))
r7 = repeat(h7,4)
h8 = (40+1*4+1):(40+2*4)
c8 = repeat(h8,inner=(4,))
r8 = repeat(h8,4)
rows = vcat(r1,r2,r3,r4,r5,r6,r7,r8)
cols = vcat(c1,c2,c3,c4,c5,c6,c7,c8)

v = vcat(vals...)
v = vcat(v,v)

@test all(matrix.rows .== rows)
@test all(matrix.cols .== cols)
@test all(matrix.vals .≈ v)



rhs = HDGElasticity.SystemRHS()
@test isempty(rhs.rows)
@test isempty(rhs.vals)
rows = [1,2,3,4]
vals = [1.0,2.0]
@test_throws AssertionError HDGElasticity.update!(rhs,rows,vals)
rows = [1,2,3,4]
vals = [0.5,0.25,0.1,0.3]
HDGElasticity.update!(rhs,rows,vals)
@test all(rhs.rows .== rows)
@test all(rhs.vals .== vals)



x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [1,1]
mesh = UniformMesh(x0,widths,nelements)
matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(4,4)) for i = 1:4]
HDGElasticity.assemble_hybrid_operator!(matrix,vals,mesh,20,2)
@test isempty(matrix.rows)
@test isempty(matrix.cols)
@test isempty(matrix.vals)


x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)
matrix = HDGElasticity.SystemMatrix()
vals = [vec(rand(4,4)) for i = 1:4]
HDGElasticity.assemble_hybrid_operator!(matrix,vals,mesh,40,2)
r1 = 45:48
rows = repeat(repeat(r1,4),2)
cols = repeat(repeat(r1,inner=(4,)),2)
v = [vals[2];vals[4]]
@test all(matrix.rows .== rows)
@test all(matrix.cols .== cols)
@test all(matrix.vals .== v)
