type PLSDiag{Ti<:Union{Int32,Int64}} <: PLSSolver # Sparse Choleksy solver for diagonal Λ
    L::Factor{Float64}
    RX::Base.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    ZtX::Matrix{Float64}
    ZtZ::Sparse{Float64}
    perm::Vector{Ti}
    λind::Vector
end

function PLSDiag{Tv,Ti<:Union(Int32,Int64)}(Zt::SparseMatrixCSC{Tv,Ti},X::Matrix,facs::Vector)
    if Ti ≠ SuiteSparse_long
        Zt = convert(SparseMatrixCSC{Tv,SuiteSparse_long},Zt)
    end
    ztz = Zt*Zt'
    XtX = Symmetric(X'X,:L)
    XtXdat = symcontents(XtX)
    ZtX = Zt*X
    L = cholfact(Symmetric(ztz+I,:U))
    PLSDiag(L,cholfact(XtXdat,:L),copy(ZtX),XtX,ZtX,Sparse(ztz),
            Base.SparseMatrix.CHOLMOD.get_perm(L),
            vcat([fill(j,length(ff.pool)) for (j,ff) in enumerate(facs)]...))
end

#function Base.A_ldiv_B!(s::Union(PLSDiag,PLSGeneral),uβ::Vector)
function Base.A_ldiv_B!(s::PLSDiag,uβ::Vector)
    q,p = size(s.RZX)
    length(uβ) == (p+q) || throw(DimensionMismatch(""))
    u = uβ[1:q]  # FIXME: change cholmod code to allow StridedVecOrMat and avoid creating the copy
    β = ContiguousView(uβ,q,(p,))
    if VERSION < v"0.4-"
        copy!(u,solve(s.L,permute!(u,s.perm),CHOLMOD_L))
    else
        copy!(u,CHOLMOD.solve(CHOLMOD.CHOLMOD_L,s.L,Dense(permute!(u,s.perm))))
    end
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,u,1.,β))
    if VERSION < v"0.4-"
        copy!(ContiguousView(uβ,(q,)),
              ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,β,1.,u),
                              CHOLMOD_Lt),s.perm))
    else
        copy!(ContiguousView(uβ,(q,)),
              ipermute!(convert(Matrix,CHOLMOD.solve(CHOLMOD.CHOLMOD_Lt,s.L,
                                                     Dense(BLAS.gemv!('N',-1.,s.RZX,β,1.,u)))),
                        s.perm))
    end
    uβ
end

function update!(s::PLSDiag,λ::Vector)
    for ll in λ
        isa(ll,PDScalF) || error("λ must be a vector PDScalF objects")
    end
    λvec = [ll.s::Float64 for ll in λ][s.λind]
    Base.SparseMatrix.CHOLMOD.update!(s.L,chm_scale!(copy(s.ZtZ),λvec,Sym);shift=1.)
    if VERSION < v"0.4-"
        copy!(s.RZX,solve(s.L, scale(λvec, s.ZtX)[s.perm,:], CHOLMOD_L))
    else
        copy!(s.RZX,CHOLMOD.solve(CHOLMOD.CHOLMOD_L, s.L, Dense(scale(λvec, s.ZtX)[s.perm,:])))
    end
    XtXdat = symcontents(s.XtX)
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(chfac(s.RX),XtXdat)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end
