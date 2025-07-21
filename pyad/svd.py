import numpy as np

def svd(A):
    # compute svd using jacobi rotations
    n, m = A.shape
    
    M = np.copy(A)
    V = np.eye(m)
    
    err = np.inf
    while err > 1e-2:
        local_err = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                mp = M[:, i]
                mq = M[:, j]
                theta = 0.5 * np.arctan2(2 * np.dot(mp, mq), np.dot(mp, mp) - np.dot(mq, mq))
                c = np.cos(theta)
                s = np.sin(theta)
                J = np.eye(m)
                J[i, i] = c
                J[j, j] = c
                J[i, j] = s
                J[j, i] = -s
                M = np.dot(M, J)
                V = np.dot(V, J)
                
                local_err += abs(theta)
        err = local_err / (m * (m - 1) / 2)
    
    M_norm = np.linalg.norm(M, axis=0)
    U = M / M_norm
    S = np.diag(M_norm)
    return U, S, V.T

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U_, S_, Vt_ = svd(A)
U, S, Vt = np.linalg.svd(A)
print(U_)
print(U)
print(S_)
print(S)
print(Vt_)
print(Vt)
    
            