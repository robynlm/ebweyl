"""This module provides classes & functions in order to decompose the Weyl tensor.

The class that provides the variables of interest is : Weyl.
The others provide/apply finite difference schemes to compute spatial derivatives.
The functions at the end are used to compute the metric with indices up.

Copyright (C) 2022  Robyn L. Munoz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at : robyn.munoz@yahoo.fr
"""

import numpy as np

###################################################################################
# Finite differencing schemes.
###################################################################################


def fd4_backward(f, i, inverse_dx):
    """4th order backward finite difference scheme."""
    return ((25/12)*f[i]
            + (-4)*f[i-1]
            + (3)*f[i-2]
            + (-4/3)*f[i-3]
            + (1/4)*f[i-4]) * inverse_dx
    

def fd4_centered(f, i, inverse_dx):
    """4th order centered finite difference scheme."""
    return ((1/12)*f[i-2]
            + (-2/3)*f[i-1]
            + (2/3)*f[i+1]
            + (-1/12)*f[i+2]) * inverse_dx
    

def fd4_forward(f, i, inverse_dx):
    """4th order forward finite difference scheme."""
    return ((-25/12)*f[i]
            + (4)*f[i+1]
            + (-3)*f[i+2]
            + (4/3)*f[i+3]
            + (-1/4)*f[i+4]) * inverse_dx
    

def fd6_backward(f, i, inverse_dx):
    """6th order backward finite difference scheme."""
    return ((49/20)*f[i]
            + (-6)*f[i-1]
            + (15/2)*f[i-2]
            + (-20/3)*f[i-3]
            + (15/4)*f[i-4]
            + (-6/5)*f[i-5]
            + (1/6)*f[i-6]) * inverse_dx
    
    
def fd6_centered(f, i, inverse_dx):
    """6th order centered finite difference scheme."""
    return ((-1/60)*f[i-3]
            + (3/20)*f[i-2]
            + (-3/4)*f[i-1]
            + (3/4)*f[i+1]
            + (-3/20)*f[i+2]
            + (1/60)*f[i+3]) * inverse_dx
    

def fd6_forward(f, i, inverse_dx):
    """6th order forward finite difference scheme."""
    return ((-49/20)*f[i]
            + (6)*f[i+1]
            + (-15/2)*f[i+2]
            + (20/3)*f[i+3]
            + (-15/4)*f[i+4]
            + (6/5)*f[i+5]
            + (-1/6)*f[i+6]) * inverse_dx
    
    
###################################################################################
# Finite differencing class applying the schemes to data box.
###################################################################################


class FiniteDifference():
    """This class applies the FD schemes to the entire data box."""
    def __init__(self, dx, N, periodic_boundary=True, fd_order6=False):
        """Define how FD schemes are to be applied. Define FDscheme.
        
        Parameters : 
            dx : float, elementary grid size
                 Here I assume dx = dy = dz
            N : int, number of data points in the box
                Here I assume that there is the same number of points 
                in all 3 directions.
            periodic_boundary : boolean, default True
                                if true periodic boundaries are applied and a 
                                centered FD scheme is used, 
                                else a combination of forward + centered + backward 
                                FD schemes are used.
            fd_order6 : boolean, default False
                        if true 6th order FD schemes are applied
                        else 4th FD schemes are applied
        """
        self.N = N
        self.periodic_boundary = periodic_boundary
        self.inverse_dx = 1 / dx
        
        if fd_order6:
            self.backward = fd6_backward
            self.centered = fd6_centered
            self.forward = fd6_forward
            self.mask_len = 3
        else:
            self.backward = fd4_backward
            self.centered = fd4_centered
            self.forward = fd4_forward
            self.mask_len = 2
    
    def d3x(self, f): 
        """Derivative along x of a scalar over a 3D box: \partial_x (f)."""
        if self.periodic_boundary:
            # Periodic boundaries are used.
            # The box is extended along the x direction by the 
            # FD mask number of points from the opposite edge.
            flong = np.concatenate((f[-self.mask_len:, :, :], f, 
                                    f[:self.mask_len, :, :]), axis=0)
            # excluding the edge points.  We retrieve shape (N, N, N).
            return np.array([self.centered(flong, ix, self.inverse_dx) 
                             for ix in range(self.mask_len, self.N+self.mask_len)])
        else:
            # There are no periodic boundaries so a combination
            # of backward centered and forward schemes are used.
            # lhs : Apply the forward FD scheme to the edge points in the x
            # direction that can not use the centered FD scheme.
            lhs = np.array([self.forward(f, ix, self.inverse_dx) 
                            for ix in range(0, self.mask_len)])
            # Apply the centered FD scheme to all points not affected
            # by the boundary condition.
            central_part = np.array([self.centered(f, ix, self.inverse_dx) 
                                     for ix in range(self.mask_len, 
                                                     self.N-self.mask_len)])
            # rhs : Apply the forward FD scheme to the edge points in the x
            # direction that can not use the centered FD scheme.
            rhs = np.array([self.backward(f, ix, self.inverse_dx) 
                            for ix in range(self.N-self.mask_len, self.N)])
            # Concatenate all the points together
            return np.concatenate((lhs, central_part, rhs), axis=0)
    
    def d3y(self, f):  
        """Derivative along y of a scalar over a 3D box: \partial_y (f)."""
        # Same as D3x but as we apply the FD schemes in the y direction 
        # we loop over the x direction to access it.
        if self.periodic_boundary:
            flong = np.concatenate((f[:, -self.mask_len:, :], f, 
                                    f[:, :self.mask_len, :]), axis=1)
            return np.array([[self.centered(flong[ix, :, :], iy, self.inverse_dx) 
                              for iy in range(self.mask_len, self.N+self.mask_len)] 
                             for ix in range(self.N)])
        else:
            lhs = np.array([[self.forward(f[ix, :, :], iy, self.inverse_dx) 
                             for iy in range(0, self.mask_len)] 
                            for ix in range(self.N)])
            central_part = np.array([[self.centered(f[ix, :, :], iy, 
                                                    self.inverse_dx) 
                                      for iy in range(self.mask_len, 
                                                      self.N-self.mask_len)] 
                                     for ix in range(self.N)])
            rhs = np.array([[self.backward(f[ix, :, :], iy, self.inverse_dx) 
                             for iy in range(self.N-self.mask_len, self.N)] 
                            for ix in range(self.N)])
            return np.concatenate((lhs, central_part, rhs), axis=1)
    
    def d3z(self, f):  
        """Derivative along z of a scalar over a 3D box: \partial_z (f)."""
        # Same as D3x but as we apply the FD schemes in the z direction 
        # we loop over the x and y directions to access it.
        if self.periodic_boundary:
            flong = np.concatenate((f[:, :, -self.mask_len:], f, 
                                    f[:, :, :self.mask_len]), axis=2)
            return np.array([[[self.centered(flong[ix, iy, :], iz, self.inverse_dx) 
                               for iz in range(self.mask_len, 
                                               self.N+self.mask_len)] 
                              for iy in range(self.N)] 
                             for ix in range(self.N)])
        else:
            lhs = np.array([[[self.forward(f[ix, iy, :], iz, self.inverse_dx) 
                              for iz in range(0, self.mask_len)] 
                             for iy in range(self.N)] 
                            for ix in range(self.N)])
            central_part = np.array([[[self.centered(f[ix, iy, :], iz, 
                                                     self.inverse_dx) 
                                       for iz in range(self.mask_len, 
                                                       self.N-self.mask_len)] 
                                      for iy in range(self.N)] 
                                     for ix in range(self.N)])
            rhs = np.array([[[self.backward(f[ix, iy, :], iz, self.inverse_dx) 
                              for iz in range(self.N-self.mask_len, self.N)] 
                             for iy in range(self.N)] for ix in range(self.N)])
            return np.concatenate((lhs, central_part, rhs), axis=2)
    
    
    def d3_scalar(self, f):
        """Spatial derivatives of a scalar: \partial_i (f)."""
        return np.array([self.d3x(f), self.d3y(f), self.d3z(f)])
    
    
    def d3_rank2tensor(self, f):
        """Spatial derivatives of a spatial rank 2 tensor"""
        #\partial_i (f_{kj}) or \partial_i (f^{kj})."""
        return np.array([[[self.d3x(f[k, j]) 
                           for j in range(3)] for k in range(3)],
                         [[self.d3y(f[k, j]) 
                           for j in range(3)] for k in range(3)],
                         [[self.d3z(f[k, j]) 
                           for j in range(3)] for k in range(3)]])
    
    def cutoffmask(self, f):
        """Remove points affected by the boundary condition."""
        return f[2*self.mask_len:-2*self.mask_len, 
                 2*self.mask_len:-2*self.mask_len, 
                 2*self.mask_len:-2*self.mask_len]
    
    
###################################################################################
# Class calculating the 3Ricci and the electric and magnetic parts of the 
# Weyl tensor as well as the Petrov scalar invariants.
###################################################################################


class Weyl():
    """Class to compute 3+1 terms, Weyl tensor and its decompositions.
    
    This class provides the variables of the 3+1 foliation of spacetime 
    for a given metric and its extrinsic curvature. It then provides
    the necessary computations to obtain the electric and magnetic parts 
    of the Weyl tensor and the invariant scalars used for the 
    Petrov classification of spacetime.
    """
    def __init__(self, FD, gdown4, Kdown4):
        """Compute 3+1 terms needed in the computations provided by this class.
        
        Parameters : 
            FD : Finite difference class
            gdown4 : (4, 4, N, N, N) array_like
                     Spacetime metric with both indices down
                     in every position of the data box.
            Kdown4 : (4, 4, N, N, N) array_like
                     Extrinsic curvature with both indices down
                     in every position of the data box.
        """
        self.FD = FD
        
        # Define 3+1 terms from the metric.
        # Spacetime metric
        self.gdown4 = gdown4
        self.gup4 = inverse4(self.gdown4)
        # Spatial metric
        self.gammadown3 = gdown4[1:, 1:]
        self.gammaup3 = inverse3(self.gammadown3)
        # Shift
        self.betadown3 = np.array([gdown4[0, 1], gdown4[0, 2], gdown4[0, 3]])
        # Lapse
        self.alpha = np.sqrt(np.einsum('k..., i..., ki... -> ...', self.betadown3, 
                                       self.betadown3, self.gammaup3) 
                             - gdown4[0, 0])
        Box_0 = np.zeros(np.shape(self.alpha))
        # Normal to the hypersurface
        self.ndown4 = np.array([-self.alpha, Box_0, Box_0, Box_0])
        self.nup4 = np.einsum('a..., ab... -> b...', self.ndown4, self.gup4)
        # Extrinsic curvature
        self.Kdown3 = Kdown4[1:,1:]
        
    def christoffel_symbol_udd3(self):
        """Compute Christoffel symbols for the spatial metric.
        
        Returns : 
            (3, 3, 3, N, N, N) array_like 
            With one indice up and two down : \Gamma^{i}_{kl}
        """
        # First the spatial derivatives of the metric derivative are computed.
        dgammaxx = self.FD.d3_scalar(self.gammadown3[0, 0]) 
        #        = [dxgxx, dygxx, dzgxx]
        dgammaxy = self.FD.d3_scalar(self.gammadown3[0, 1])
        dgammaxz = self.FD.d3_scalar(self.gammadown3[0, 2])
        dgammayy = self.FD.d3_scalar(self.gammadown3[1, 1])
        dgammayz = self.FD.d3_scalar(self.gammadown3[1, 2])
        dgammazz = self.FD.d3_scalar(self.gammadown3[2, 2])
        
        # Spatial Christoffel symbols with all indices down: \Gamma_{jkl}.
        Gxyz = dgammaxz[1] + dgammaxy[2] - dgammayz[0]
        Gx = np.array([[dgammaxx[0], dgammaxx[1], dgammaxx[2]],
                       [dgammaxx[1], 2*dgammaxy[1]-dgammayy[0], Gxyz],
                       [dgammaxx[2], Gxyz, 2*dgammaxz[2]-dgammazz[0]]]) / 2
        
        Gyxz = dgammayz[0] + dgammaxy[2] - dgammaxz[1]
        Gy = np.array([[2*dgammaxy[0]-dgammaxx[1], dgammayy[0], Gyxz],
                       [dgammayy[0], dgammayy[1], dgammayy[2]],
                       [Gyxz, dgammayy[2], 2*dgammayz[2]-dgammazz[1]]]) / 2
        
        Gzxy = dgammayz[0] + dgammaxz[1] - dgammaxy[2]
        Gz = np.array([[2*dgammaxz[0]-dgammaxx[2], Gzxy, dgammazz[0]],
                       [Gzxy, 2*dgammayz[1]-dgammayy[2], dgammazz[1]],
                       [dgammazz[0], dgammazz[1], dgammazz[2]]]) / 2
        Gddd = np.array([Gx,Gy,Gz])
        
        # Spatial Christoffel symbols with indices: \Gamma^{i}_{kl}.
        Gudd3 = np.einsum('ij..., jkl... -> ikl...', self.gammaup3, Gddd)
        return Gudd3
    
    def ricci_tensor_down3(self, Gudd3):
        """Compute spatial Ricci tensor with both indices down.
        
        Parameters : 
            Gudd3 : (3, 3, 3, N, N, N) array_like 
                    Spatial Christoffel symbol with one indice up and two down
                
        Returns : 
            (3, 3, N, N, N) array_like
        
        """
        Rterm0 = np.array([[self.FD.d3x(Gudd3[0, j, k])
                            + self.FD.d3y(Gudd3[1, j, k])
                            + self.FD.d3z(Gudd3[2, j, k]) 
                            for k in range(3)] 
                           for j in range(3)])  # = \partial_i \Gamma^{i}_{jk}
        Gd3 = np.einsum('iik... -> k...', Gudd3)
        Rterm1 = np.array([self.FD.d3_scalar(Gd3[j]) for j in range(3)])
        Rterm2 = np.einsum('iip..., pjk... -> jk...', Gudd3, Gudd3)
        Rterm3 = np.einsum('ijp..., pik... -> jk...', Gudd3, Gudd3)
        Ricci3down3 = Rterm0 - Rterm1 + Rterm2 - Rterm3   #R_{jk}
        return Ricci3down3
    
    def eweyl_u_tensor_down4(self, Cdown4, uup4):
        """Compute 4D electric part of the Weyl tensor projected along u^{\mu}.
        
        u^{\mu} : chosen time-like vector
        
        Parameters : 
            Cdown4 : (4, 4, 4, 4, N, N, N) array_like
                     4D Weyl tensor with all indices down.
            uup4 : (4, N, N, N) array_like
                   Time-like vector with indice up
                
        Returns : 
            (4, 4, N, N, N) array_like
        """
        return np.einsum('b..., d..., abcd... -> ac...', uup4, uup4, Cdown4)
    
    def eweyl_n_tensor_down3(self, Ricci3down3, kappa, Tdown4):
        """Compute 3D electric part of the Weyl tensor projected along n^{\mu}.
        
        n^{\mu} : the normal to the hypersurface.
        
        Parameters : 
            Ricci3down3 : (3, 3, N, N, N) array_like 
                          Spatial Ricci tensor with both indices down
            kappa : float, Einstein's constant = 8 * pi * G / c^4
            Tdown4 : (4, 4, N, N, N) array_like
                     Spacetime stress-energy tensor with both indices down
                
        Returns : 
            (3, 3, N, N, N) array_like
        """
        # 1st compute K terms
        Kmixed3 = np.einsum('ij..., jk... -> ik...', self.gammaup3, self.Kdown3)
        Ktrace = self.trace_rank2tensor3(self.Kdown3)
        KKterm = np.einsum('im..., mj... -> ij...', self.Kdown3, Kmixed3)
        KKtermH = np.einsum('ij..., ji... -> ...', Kmixed3, Kmixed3)
        del Kmixed3
        
        # 2nd compute S terrms
        gmixed4 = np.einsum('ab..., bc... -> ac...', self.gup4, self.gdown4)
        gammamixed4 = gmixed4 + np.einsum('a..., c... -> ac...',
                                          self.ndown4, self.nup4)
        Sdown3 = np.einsum('ca..., db..., cd... -> ab...', 
                           gammamixed4, gammamixed4, Tdown4)[1:,1:]
        Strace  = self.trace_rank2tensor3(Sdown3)
        del gmixed4, gammamixed4
        
        # last 3Ricci scalar
        Ricci3S = self.trace_rank2tensor3(Ricci3down3)
        
        # Now find E
        Endown3 = (Ricci3down3 + Ktrace*self.Kdown3 - KKterm 
                   - (1/3)*self.gammadown3*(Ricci3S + Ktrace*Ktrace - KKtermH) 
                   - (kappa/2)*(Sdown3 - self.gammadown3*Strace/3))
        return Endown3
    
    def bweyl_u_tensor_down4(self, Cdown4, uup4):
        """Compute 4D magnetic part of the Weyl tensor projected along u^{\mu}.
        
        u^{\mu} : chosen time-like vector
        
        Parameters : 
            Cdown4 : (4, 4, 4, 4, N, N, N) array_like
                     4D Weyl tensor with all indices down.
            uup4 : (4, N, N, N) array_like
                   Time-like vector with indice up
                
        Returns : 
            (4, 4, N, N, N) array_like
        """
        LCuudd4 = np.einsum('ac..., bd..., abef... -> cdef...', 
                            self.gup4, self.gup4, self.levicivita_tensor_down4())
        Budown4 = np.einsum('b..., f..., abcd..., cdef... -> ae...', 
                            uup4, uup4, Cdown4, LCuudd4) / 2
        return Budown4
    
    def bweyl_n_tensor_down3(self, Gudd3):
        """Compute 3D magnetic part of the Weyl tensor projected along n^{\mu}.
        
        n^{\mu} : the normal to the hypersurface.
        
        Parameters : 
            Gudd3 : (3, 3, 3, N, N, N) array_like 
                    Spatial Christoffel symbol with one indice up and two down
                
        Returns : 
            (3, 3, N, N, N) array_like
        """
        LCuud3 = np.einsum('ae..., bf..., d..., defc... -> abc...', 
                           self.gup4, self.gup4, self.nup4, 
                           self.levicivita_tensor_down4())[1:, 1:, 1:]
        
        dKdown = self.covariant_derivatice_3_tensor2down3(Gudd3, self.Kdown3)
        Bterm1 = np.einsum('cdb..., cda... -> ab...', LCuud3, dKdown)
        
        Ktrace  = self.trace_rank2tensor3(self.Kdown3)
        Kmixed3 = np.einsum('ij..., jk... -> ik...', self.gammaup3, self.Kdown3)
        Bterm2K = (self.covariant_derivatice_3_scalar(Ktrace) 
                   - np.einsum('ccb... -> b...', 
                               self.covariant_derivatice_3_tensor2mixed3(Gudd3,
                                                                         Kmixed3)))
        Bterm2 = np.einsum('cdb..., ac..., d... -> ab...', LCuud3, 
                           self.gammadown3, Bterm2K)/2
        
        Bndown3 = Bterm1 + Bterm2
        return Bndown3
    
    def ebweyl_n_3D_to_4D(self, fdown3):
        """Compute spacetime tensor from the spatial tensor.
        
        Parameters : 
            fdown3 : (3, 3, N, N, N) array_like
        
        Returns : 
            (4, 4, N, N, N) array_like
            
        Note : 
            By definition {}^{(n)}E^{\alpha\beta} only has spatial components, 
            same for the magnetic part.
        
        Warning : 
            This is only for the electric and 
            magnetic parts of the Weyl tensor projected along 
            the normal to the hypersurface.
        """
        fup3 = np.einsum('ib...,ja...,ab... -> ij...',
                         self.gammaup3, self.gammaup3, fdown3)
        fmixed3 = np.einsum('ij...,jk...->ik...', self.gammaup3, fdown3)
        f00 = np.einsum('i..., j..., ij... -> ...', self.betadown3, 
                        self.betadown3, fup3)
        f0k = np.einsum('i..., ik... -> k...', self.betadown3, fmixed3)
        fdown4 = np.array([[f00, f0k[0], f0k[1], f0k[2]],
                           [f0k[0], fdown3[0, 0], fdown3[0, 1], fdown3[0, 2]],
                           [f0k[1], fdown3[1, 0], fdown3[1, 1], fdown3[1, 2]],
                           [f0k[2], fdown3[2, 0], fdown3[2, 1], fdown3[2, 2]]])
        return fdown4
    
    def weyl_tensor_down4(self, Endown3, Bndown3): 
        """Compute Weyl tensor with all indices down.
        
        Parameters : 
            Endown3 : (3, 3, N, N, N) array_like
            Bndown3 : (3, 3, N, N, N) array_like
                      Electric and magnetic parts of the Weyl tensor 
                      projected along the normal to the hypersurface.
        
        Returns : 
            (4, 4, 4, 4, N, N, N) array_like
        
        Reference : 
            'Introduction to 3+1 Numerical Relativity' 2008
            by M. Alcubierre
            equation : 8.3.13
        """
        Endown4 = self.ebweyl_n_3D_to_4D(Endown3)
        Bndown4 = self.ebweyl_n_3D_to_4D(Bndown3)
        ldown4 = self.gdown4 + 2.0 * np.einsum('a..., b... -> ab...',
                                                   self.ndown4, self.ndown4)
        LCudd4 = np.einsum('ec..., d..., dcab... -> eab...', self.gup4, 
                           self.nup4, self.levicivita_tensor_down4())
        
        Cdown4 = (np.einsum('ac..., db... -> abcd...', ldown4, Endown4)
                  - np.einsum('ad..., cb... -> abcd...', ldown4, Endown4))
        Cdown4 -= (np.einsum('bc..., da... -> abcd...', ldown4, Endown4)
                   - np.einsum('bd..., ca... -> abcd...', ldown4, Endown4))
        Cdown4 -= np.einsum('cde..., eab... -> abcd...',
                            (np.einsum('c..., de... -> cde...',
                                       self.ndown4, Bndown4)
                             - np.einsum('d..., ce... -> cde...',
                                         self.ndown4, Bndown4)), LCudd4)
        Cdown4 -= np.einsum('abe..., ecd... -> abcd...', 
                            (np.einsum('a..., be... -> abe...', 
                                       self.ndown4, Bndown4) 
                             - np.einsum('b..., ae... -> abe...', 
                                         self.ndown4, Bndown4)), LCudd4)
        return Cdown4    
                   
    def weyl_psi_scalars(self, Cdown4):
        """Compute Weyl scalars with an arbitrary null vector base.
        
        Parameters : 
            Cdown4 : (4, 4, 4, 4, N, N, N) array_like
                     Weyl tensor with all indices down.
        
        Returns : 
            list : psi0, psi1, psi2, psi3, psi4
                   Each is (N, N, N) array_like complex
        """
        lup, kup, mup, mbup = self.null_vector_base()
        psi0 = np.einsum('abcd..., a..., b..., c..., d... -> ...', 
                         Cdown4, kup, mup, kup, mup)
        psi1 = np.einsum('abcd..., a..., b..., c..., d... -> ...', 
                         Cdown4, kup, lup, kup, mup)
        psi2 = np.einsum('abcd..., a..., b..., c..., d... -> ...', 
                         Cdown4, kup, mup, mbup, lup)
        psi3 = np.einsum('abcd..., a..., b..., c..., d... -> ...', 
                         Cdown4, kup, lup, mbup, lup)
        psi4 = np.einsum('abcd..., a..., b..., c..., d... -> ...', 
                         Cdown4, mbup, lup, mbup, lup)

        # As these are then used to compute the invariant scalars, here I check 
        # if psi4 = 0 while psi0 =/= 0.  If it is the case I need to switch
        # psi0 and psi4 as well as psi1 and psi3 so I do that here.
        mask = np.where(np.logical_and(abs(psi4) < 1e-5, abs(psi0) > 1e-5))
        psi0new = psi0
        psi0new[mask] = psi4[mask]
        psi4[mask] = psi0[mask]
        psi0 = psi0new
        psi1new = psi1
        psi1new[mask] = psi3[mask]
        psi3[mask] = psi1[mask]
        psi1 = psi1new
        return [psi0, psi1, psi2, psi3, psi4]

    def null_vector_base(self):
        """Return an arbitrary null vector base.
        
        Returns : 
            list : lup, kup, mup, mbup
                   Each is (4, N, N, N) array_like complex
        
        Reference : 
            'Introduction to 3+1 Numerical Relativity' 2008
            by M. Alcubierre
            page 295
        """
        e0, e1, e2, e3 = self.tetrad_base()
        inverse_sqrt_2 = 1 / np.sqrt(2)
        kup = (e0+e1) * inverse_sqrt_2
        lup = (e0-e1) * inverse_sqrt_2
        mup = (e2+1j*e3) * inverse_sqrt_2
        mbup = (e2-1j*e3) * inverse_sqrt_2
        return lup, kup, mup, mbup

    def tetrad_base(self):
        """Return an arbitrary orthonormal tetrad base.
        
        The first tetrad is the normal to the hypersurface.
        The others are arbitrarily chosen and made orthonormal 
        with the Gram-Schmidt scheme.
        
        Returns : 
            list : e0, e1, e2, e3
                   Each is (4, N, N, N) array_like        
        """
        Box_0 = np.zeros(np.shape(self.alpha))
        
        v1 = np.array([Box_0, 1.0/np.sqrt(self.gdown4[1,1]), Box_0, Box_0])
        v2 = np.array([Box_0, Box_0, 1.0/np.sqrt(self.gdown4[2,2]), Box_0])
        v3 = np.array([Box_0, Box_0, Box_0, 1.0/np.sqrt(self.gdown4[3,3])])

        u0 = self.nup4
        u1 = v1 - self.vector_projection4(u0, v1)
        u2 = (v2 - self.vector_projection4(u0, v2) 
              - self.vector_projection4(u1, v2))
        u3 = (v3 - self.vector_projection4(u0, v3) 
              - self.vector_projection4(u1, v3) 
              - self.vector_projection4(u2, v3))

        e0 = u0 / self.norm_rank1tensor4(u0)
        e1 = u1 / self.norm_rank1tensor4(u1)
        e2 = u2 / self.norm_rank1tensor4(u2)
        e3 = u3 / self.norm_rank1tensor4(u3)

        return e0, e1, e2, e3
    
    def invariant_scalars(self, Psis):
        """Compute scalar invariants used for Petrov classification.
        
        Parameters : 
            Psis : [psi0, psi1, psi2, psi3, psi4]
                   Each is a (N, N, N) array_like complex
        
        Returns : 
            dictionary : I, J, L, K, N
                         Each is a (N, N, N) array_like complex.
        
        Reference : 
            'Exact Solutions to Einstein's Field Equations' 2nd edition 2003 
            by H. Stephani, D. Kramer,  M. A. H. MacCallum, C. Hoenselaers
        """
        I_inv = Psis[0]*Psis[4] - 4*Psis[1]*Psis[3] + 3*Psis[2]*Psis[2]
        J_inv = determinant3(np.array([[Psis[4], Psis[3], Psis[2]], 
                                       [Psis[3], Psis[2], Psis[1]], 
                                       [Psis[2], Psis[1], Psis[0]]]))
        L_inv = Psis[2]*Psis[4] - (Psis[3]**2)
        K_inv = Psis[1]*(Psis[4]**2) - 3*Psis[4]*Psis[3]*Psis[2] + 2*(Psis[3]**3)
        N_inv = 12*(L_inv**2) - (Psis[4]**2)*I_inv
        return {'I': I_inv, 'J': J_inv, 'L': L_inv, 'K': K_inv, 'N': N_inv}
        
    def covariant_derivatice_3_scalar(self, f):
        """Compute spatial covariant derivative of a scalar.
        
        Covariant derivative with respects to the spatial metric.
        
        Parameters : 
            f : (N, N, N) array_like
        
        Returns : 
            (3, N, N, N) array_like
             ^-- new indice from the derivation
        """
        return self.FD.d3_scalar(f)

    def covariant_derivatice_3_tensor2down3(self, Gudd3, fdown3):
        """Compute spatial covariant derivative of a 3D rank 2 covariant tensor.
        
        Covariant derivative with respects to the spatial metric.
        
        Parameters : 
            Gudd3 : (3, 3, 3, N, N, N) array_like 
                    Spatial Christoffel symbol with one indice up and two down
            fdown3 : (3, 3, N, N, N) array_like
                     Rank 2 spatial tensor with both indices down
        
        Returns : 
            (3, 3, 3, N, N, N) array_like
                ^--^-- fdown3 indices
             ^-- new indice from the derivation
        """
        df = self.FD.d3_rank2tensor(fdown3)
        G1 = - np.einsum('dca..., db... -> cab...', Gudd3, fdown3)
        G2 = - np.einsum('dcb..., ad... -> cab...', Gudd3, fdown3)
        return df + G1 + G2
    
    def covariant_derivatice_3_tensor2mixed3(self, Gudd3, fmixed3):
        """Compute spatial covariant derivative of a 3D rank 2 mixed indice tensor.
        
        Covariant derivative with respects to the spatial metric.
        
        Parameters : 
            Gudd3 : (3, 3, 3, N, N, N) array_like 
                    Spatial Christoffel symbol with one indice up and two down
            fmixed3 : (3, 3, N, N, N) array_like
                      Rank 2 spatial tensor with one indice up and the other down
        
        Returns : 
            (3, 3, 3, N, N, N) array_like
                ^--^-- fmixed3 indices
             ^-- new indice from the derivation
        """
        df = self.FD.d3_rank2tensor(fmixed3)
        G1 = np.einsum('acd..., db... -> cab...', Gudd3, fmixed3)
        G2 = - np.einsum('dcb..., ad... -> cab...', Gudd3, fmixed3)
        return df + G1 + G2

    def vector_projection4(self, a, b): 
        """Project vector b onto vector a."""        
        return (np.einsum('a..., b..., ab... -> ...', a, b, self.gdown4) 
                * a / np.einsum('a..., b..., ab... -> ...', a, a, self.gdown4))

    def trace_rank2tensor3(self, fdown3):
        """Compute trace of a 3D rank 2 tensor."""
        return np.einsum('jk..., jk... -> ...', self.gammaup3, fdown3)
    
    def norm_rank1tensor4(self, a): 
        """Compute norm of a 4D rank 1 tensor."""
        return np.sqrt(abs(np.einsum('a..., b..., ab... -> ...',a,a,self.gdown4)))
    
    def norm_rank2tensor3(self, fdown3):
        """Compute norm of a 3D rank 2 tensor."""
        fup3 = np.einsum('ib..., ja..., ab... -> ij...', self.gammaup3,
                         self.gammaup3, fdown3)
        return np.sqrt(abs(np.einsum('ab..., ab... -> ...', fup3, fdown3)))
    
    def norm_rank2tensor4(self, fdown4):
        """Compute norm of a 4D rank 2 tensor."""
        fup4 = np.einsum('ib..., ja..., ab... -> ij...', self.gup4,
                         self.gup4, fdown4)
        return np.sqrt(abs(np.einsum('ab..., ab... -> ...', fup4, fdown4)))
    
    def levicivita_tensor_down4(self):
        """Compute spacetime Levi-Civita tensor with 4 4D indices down."""
        return (self.levicivita_symbol_down4() 
                * np.sqrt(abs(determinant4(self.gdown4))))

    def levicivita_symbol_down4(self): 
        """Compute spacetime Levi-Civita symbol with 4 4D indices down."""
        LC = np.zeros((4, 4, 4, 4, self.FD.N, self.FD.N, self.FD.N))
        allindices = [0, 1, 2, 3]
        for i0 in allindices:
            for i1 in np.delete(allindices, i0):
                for i2 in np.delete(allindices, [i0, i1]):
                    for i3 in np.delete(allindices, [i0, i1, i2]):
                        top = ((i1-i0) * (i2-i0) * (i3-i0) 
                               * (i2-i1) * (i3-i1) * (i3-i2))
                        bot = (abs(i1-i0) * abs(i2-i0) * abs(i3-i0)
                               * abs(i2-i1) * abs(i3-i1) * abs(i3-i2))
                        LC[i0, i1, i2, i3, :, :, :] = float(top/bot)
        return LC
    
                   
###################################################################################
# Some useful tools.
###################################################################################
   
                   
def getcomponents3(f):
    """Extract components of a rank 2 tensor with 3D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters : 
        f : (3, 3, N, N, N) array_like
    
    Returns : 
        list : xx, xy, xz, yy, yz, zz
               Each is (N, N, N) array_like
    """
    return [f[0, 0], f[0, 1], f[0, 2], f[1, 1], f[1, 2], f[2, 2]]
                   
                   
def getcomponents4(f):
    """Extract components of a rank 2 tensor with 4D indices.
    
    This assumes this tensor is symmetric.
    
    Parameters : 
        f : (4, 4, N, N, N) array_like
    
    Returns : 
        list : tt, tx, ty, tz, xx, xy, xz, yy, yz, zz
               Each is (N, N, N) array_like
    """
    return [f[0, 0], f[0, 1], f[0, 2], f[0, 3], 
            f[1, 1], f[1, 2], f[1, 3], 
            f[2, 2], f[2, 3], f[3, 3]]
    
                   
def determinant3(f):
    """Compute determinant 3x3 matrice in every position of the data box."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    return -xz*xz*yy + 2*xy*xz*yz - xx*yz*yz - xy*xy*zz + xx*yy*zz
                   
                   
def determinant4(f):
    """Compute determinant of a 4x4 matrice in every position of the data box."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    return (tz*tz*xy*xy - 2*ty*tz*xy*xz + ty*ty*xz*xz 
            - tz*tz*xx*yy + 2*tx*tz*xz*yy - tt*xz*xz*yy 
            + 2*ty*tz*xx*yz - 2*tx*tz*xy*yz - 2*tx*ty*xz*yz 
            + 2*tt*xy*xz*yz + tx*tx*yz*yz - tt*xx*yz*yz 
            - ty*ty*xx*zz + 2*tx*ty*xy*zz - tt*xy*xy*zz 
            - tx*tx*yy*zz + tt*xx*yy*zz)

                   
def inverse3(f):
    """Compute inverse of a 3x3 matrice in every position of the data box."""
    xx, xy, xz, yy, yz, zz = getcomponents3(f)
    fup = np.array([[yy*zz - yz*yz, -(xy*zz - yz*xz), xy*yz - yy*xz], 
                    [-(xy*zz - xz*yz), xx*zz - xz*xz, -(xx*yz - xy*xz)],
                    [xy*yz - xz*yy, -(xx*yz - xz*xy), xx*yy - xy*xy]])
    return fup / determinant3(f)
                   
                   
def inverse4(f):
    """Compute inverse of a 4x4 matrice in every position of the data box."""
    tt, tx, ty, tz, xx, xy, xz, yy, yz, zz = getcomponents4(f)
    fup = np.array([[-xz*xz*yy + 2*xy*xz*yz - xx*yz*yz - xy*xy*zz + xx*yy*zz, 
                     (tz*xz*yy - tz*xy*yz - ty*xz*yz 
                      + tx*yz*yz + ty*xy*zz - tx*yy*zz), 
                     (-tz*xy*xz + ty*xz*xz + tz*xx*yz 
                      - tx*xz*yz - ty*xx*zz + tx*xy*zz), 
                     (tz*xy*xy - ty*xy*xz - tz*xx*yy 
                      + tx*xz*yy + ty*xx*yz - tx*xy*yz)], 
                    [(tz*xz*yy - tz*xy*yz - ty*xz*yz 
                      + tx*yz*yz + ty*xy*zz - tx*yy*zz), 
                     -tz*tz*yy + 2*ty*tz*yz - tt*yz*yz - ty*ty*zz + tt*yy*zz,
                     (tz*tz*xy - ty*tz*xz - tx*tz*yz 
                      + tt*xz*yz + tx*ty*zz - tt*xy*zz), 
                     (-ty*tz*xy + ty*ty*xz + tx*tz*yy 
                      - tt*xz*yy - tx*ty*yz + tt*xy*yz)], 
                    [(-tz*xy*xz + ty*xz*xz + tz*xx*yz 
                      - tx*xz*yz - ty*xx*zz + tx*xy*zz), 
                     (tz*tz*xy - ty*tz*xz - tx*tz*yz 
                      + tt*xz*yz + tx*ty*zz - tt*xy*zz), 
                     -tz*tz*xx + 2*tx*tz*xz - tt*xz*xz - tx*tx*zz + tt*xx*zz, 
                     (ty*tz*xx - tx*tz*xy - tx*ty*xz 
                      + tt*xy*xz + tx*tx*yz - tt*xx*yz)], 
                    [(tz*xy*xy - ty*xy*xz - tz*xx*yy 
                      + tx*xz*yy + ty*xx*yz - tx*xy*yz), 
                     (-ty*tz*xy + ty*ty*xz + tx*tz*yy 
                      - tt*xz*yy - tx*ty*yz + tt*xy*yz), 
                     (ty*tz*xx - tx*tz*xy - tx*ty*xz 
                      + tt*xy*xz + tx*tx*yz - tt*xx*yz), 
                     -ty*ty*xx + 2*tx*ty*xy - tt*xy*xy - tx*tx*yy + tt*xx*yy]])
    return fup / determinant4(f)
    