"""
This module performs subspace system identification.

It enforces that matrices are used instead of arrays
to avoid dimension conflicts.
"""
import pylab as pl
from . import ss

__all__ = ['subspace_det_algo1', 'prbs', 'nrms']

#pylint: disable=invalid-name

def block_hankel(data, f):
    """
    Create a block hankel matrix.
    f : number of rows
    """
    data = pl.matrix(data)
    assert len(data.shape) == 2
    n = data.shape[1] - f
    return pl.matrix(pl.hstack([
        pl.vstack([data[:, i+j] for i in range(f)])
        for j in range(n)]))

def project(A):
    """
    Creates a projection matrix onto the rowspace of A.
    """
    A = pl.matrix(A)
    return  A.T*(A*A.T).I*A

def project_perp(A):
    """
    Creates a projection matrix onto the space perpendicular to the
    rowspace of A.
    """
    A = pl.matrix(A)
    I = pl.matrix(pl.eye(A.shape[1]))
    P = project(A)
    return  I - P

def project_oblique(B, C):
    """
    Projects along rowspace of B onto rowspace of C.
    """
    proj_B_perp = project_perp(B)
    return proj_B_perp*(C*proj_B_perp).I*C

def subspace_det_algo1(y, u, f, p, s_tol, dt):
    """
    Subspace Identification for deterministic systems
    deterministic algorithm 1 from (1)

    assuming a system of the form:

    x(k+1) = A x(k) + B u(k)
    y(k)   = C x(k) + D u(k)

    and given y and u.

    Find A, B, C, D

    See page 52. of (1)

    (1) Subspace Identification for Linear
    Systems, by Van Overschee and Moor. 1996
    """
    #pylint: disable=too-many-arguments, too-many-locals
    # for this algorithm, we need future and past
    # to be more than 1
    assert f > 1
    assert p > 1

    # setup matrices
    y = pl.matrix(y)
    n_y = y.shape[0]
    u = pl.matrix(u)
    n_u = u.shape[0]
    w = pl.vstack([y, u])
    n_w = w.shape[0]

    # make sure the input is column vectors
    assert y.shape[0] < y.shape[1]
    assert u.shape[0] < u.shape[1]

    W = block_hankel(w, f + p)
    U = block_hankel(u, f + p)
    Y = block_hankel(y, f + p)

    W_p = W[:n_w*p, :]
    W_pp = W[:n_w*(p+1), :]

    Y_f = Y[n_y*f:, :]
    U_f = U[n_y*f:, :]

    Y_fm = Y[n_y*(f+1):, :]
    U_fm = U[n_u*(f+1):, :]

    # step 1, calculate the oblique projections
    #------------------------------------------
    # Y_p = G_i Xd_p + Hd_i U_p
    # After the oblique projection, U_p component is eliminated,
    # without changing the Xd_p component:
    # Proj_perp_(U_p) Y_p = W1 O_i W2 = G_i Xd_p
    O_i = Y_f*project_oblique(U_f, W_p)
    O_im = Y_fm*project_oblique(U_fm, W_pp)

    # step 2, calculate the SVD of the weighted oblique projection
    #------------------------------------------
    # given: W1 O_i W2 = G_i Xd_p
    # want to solve for G_i, but know product, and not Xd_p
    # so can only find Xd_p up to a similarity transformation
    W1 = pl.matrix(pl.eye(O_i.shape[0]))
    W2 = pl.matrix(pl.eye(O_i.shape[1]))
    U0, s0, VT0 = pl.svd(W1*O_i*W2)  #pylint: disable=unused-variable

    # step 3, determine the order by inspecting the singular
    #------------------------------------------
    # values in S and partition the SVD accordingly to obtain U1, S1
    #print s0
    n_x = pl.find(s0/s0.max() > s_tol)[-1] + 1
    U1 = U0[:, :n_x]
    # S1 = pl.matrix(pl.diag(s0[:n_x]))
    # VT1 = VT0[:n_x, :n_x]

    # step 4, determine Gi and Gim
    #------------------------------------------
    G_i = W1.I*U1*pl.matrix(pl.diag(pl.sqrt(s0[:n_x])))
    G_im = G_i[:-n_y, :]

    # step 5, determine Xd_ip and Xd_p
    #------------------------------------------
    # only know Xd up to a similarity transformation
    Xd_i = G_i.I*O_i
    Xd_ip = G_im.I*O_im

    # step 6, solve the set of linear eqs
    # for A, B, C, D
    #------------------------------------------
    Y_ii = Y[n_y*p:n_y*(p+1), :]
    U_ii = U[n_u*p:n_u*(p+1), :]

    a_mat = pl.matrix(pl.vstack([Xd_ip, Y_ii]))
    b_mat = pl.matrix(pl.vstack([Xd_i, U_ii]))
    ss_mat = a_mat*b_mat.I
    A_id = ss_mat[:n_x, :n_x]
    B_id = ss_mat[:n_x, n_x:]
    assert B_id.shape[0] == n_x
    assert B_id.shape[1] == n_u
    C_id = ss_mat[n_x:, :n_x]
    assert C_id.shape[0] == n_y
    assert C_id.shape[1] == n_x
    D_id = ss_mat[n_x:, n_x:]
    assert D_id.shape[0] == n_y
    assert D_id.shape[1] == n_u

    if pl.matrix_rank(C_id) == n_x:
        T = C_id.I # try to make C identity, want it to look like state feedback
    else:
        T = pl.matrix(pl.eye(n_x))

    Q_id = pl.zeros((n_x, n_x))
    R_id = pl.zeros((n_y, n_y))
    sys = ss.StateSpaceDiscreteLinear(
        A=T.I*A_id*T, B=T.I*B_id, C=C_id*T, D=D_id,
        Q=Q_id, R=R_id, dt=dt)
    return sys


def nrms(data_fit, data_true):
    """
    Normalized root mean square error.
    """
    # root mean square error
    rms = pl.mean(pl.norm(data_fit - data_true, axis=0))

    # normalization factor is the max - min magnitude, or 2 times max dist from mean
    norm_factor = 2*pl.norm(data_true - pl.mean(data_true, axis=1), axis=0).max()
    return (norm_factor - rms)/norm_factor

def prbs(n):
    """
    Pseudo random binary sequence.
    """
    return pl.where(pl.rand(n) > 0.5, 0, 1)

def robust_combined_algo(y, u, f, p, s_tol, dt):
    """
    Subspace Identification for stochastic systems with input
    Robust combined algorithm from chapter 4 of (1)

    assuming a system of the form:

    x(k+1) = A x(k) + B u(k) + w(k)
    y(k)   = C x(k) + D u(k) + v(k)
    E[(w_p; v_p) (w_q^T v_q^T)] = (Q S; S^T R) delta_pq

    and given y and u.

    Find the order of the system and A, B, C, D, Q, S, R

    See page 131, and generally chapter 4, of (1)
    A different implementation of the algorithm is presented in 6.1 of (1)

    (1) Subspace Identification for Linear
    Systems, by Van Overschee and Moor. 1996
    """
    #pylint: disable=too-many-arguments, too-many-locals
    # for this algorithm, we need future and past
    # to be more than 1
    assert f > 1
    assert p > 1

    # setup matrices
    y = pl.matrix(y)
    n_y = y.shape[0]
    u = pl.matrix(u)
    n_u = u.shape[0]
    w = pl.vstack([y, u])
    n_w = w.shape[0]

    # make sure the input is column vectors
    assert y.shape[0] < y.shape[1]
    assert u.shape[0] < u.shape[1]

    W = block_hankel(w, f + p)
    U = block_hankel(u, f + p)
    Y = block_hankel(y, f + p)

    W_p = W[:n_w*p, :]
    W_pp = W[:n_w*(p+1), :]

    Y_f = Y[n_y*f:, :]
    U_f = U[n_y*f:, :]

    Y_fm = Y[n_y*(f+1):, :]
    U_fm = U[n_u*(f+1):, :]

    # step 1, calculate the oblique and orthogonal projections
    #------------------------------------------
    #TODO fix explanation
    # Y_p = G_i Xd_p + Hd_i U_p
    # After the oblique projection, U_p component is eliminated,
    # without changing the Xd_p component:
    # Proj_perp_(U_p) Y_p = W1 O_i W2 = G_i Xd_p
    O_i  = Y_f*project_oblique(U_f, W_p)
    Z_i  = Y_f*project(pl.vstack(W_p, U_f))
    Z_ip = Y_fm*project(pl.vstack(W_pp, U_fm))

    #TODO fix explanation
    # step 2, calculate the SVD of the weighted oblique projection
    #------------------------------------------
    # given: W1 O_i W2 = G_i Xd_p
    # want to solve for G_i, but know product, and not Xd_p
    # so can only find Xd_p up to a similarity transformation
    U0, s0, VT0 = pl.svd(O_i*project_perp(U_f))  #pylint: disable=unused-variable

    # step 3, determine the order by inspecting the singular
    #------------------------------------------
    # values in S and partition the SVD accordingly to obtain U1, S1
    #print s0
    n_x = pl.find(s0/s0.max() > s_tol)[-1] + 1
    U1 = U0[:, :n_x]
    S1 = pl.matrix(pl.diag(s0[:n_x]))
    # VT1 = VT0[:n_x, :n_x]

    # step 4, determine Gi and Gim
    #------------------------------------------
    G_i = U1*pl.matrix(pl.diag(pl.sqrt(s1[:n_x])))
    G_im = G_i[:-n_y, :]

    # step 5, solve the linear equations for A and C
    #------------------------------------------
    # Recompute G_i and G_im from A and C
    #TODO figure out what K (contains B and D) and the rhos (residuals) are in terms of knowns
    AC_stack = (pl.vstack(G_im.I*Z_ip,Y_f(1,:))-K*U_f-pl.vstack(rho_w, rho_v))*(G_i.I*Z_i).I #TODO not done

    # step 6, Solve for B and D
    #------------------------------------------
    #TODO do minimization problem

    # step 7, determine the covariance matrices Q, S, and R
    #-------------------------------------------
    #TODO once rhos are solved for, find their covariance

# vim: set et fenc=utf-8 ft=python  ff=unix sts=4 sw=4 ts=4 :
