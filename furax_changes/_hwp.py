import equinox
import jax
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree
import jax.numpy as jnp

from furax import AbstractLinearOperator, diagonal
from furax.core.rules import AbstractBinaryRule

from ..stokes import (
    Stokes,
    StokesI,
    StokesIQU,
    StokesIQUV,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)
from ._qu_rotations import QURotationOperator, QURotationTransposeOperator


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate."""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        hwp = cls(in_structure)
        if angles is None:
            return hwp
        rot = QURotationOperator(angles, in_structure)
        rotated_hwp: AbstractLinearOperator = rot.T @ hwp @ rot
        return rotated_hwp

    def mv(self, x: StokesPyTreeType) -> Stokes:
        if isinstance(x, StokesI):
            return x
        if isinstance(x, StokesQU):
            return StokesQU(x.q, -x.u)
        if isinstance(x, StokesIQU):
            return StokesIQU(x.i, x.q, -x.u)
        if isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, x.q, -x.u, -x.v)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

mm = 1e-3
GHz = 1e9
deg = np.pi / 180.0
c = 3e8
thicknesses   = [0.394*mm, 0.04*mm, 0.212*mm, 3.75*mm,3.75*mm,3.75*mm, 0.212*mm, 0.04*mm, 0.394*mm]
thicknesses_HF = [0.183*mm, 0.04*mm, 0.097*mm, 1.60*mm,1.60*mm,1.60*mm, 0.097*mm, 0.04*mm, 0.183*mm]
angles      = [0.0, 0.0, 0.0, 0.0, 54.0*deg, 0.0, 0.0, 0.0, 0.0]


def get_delta(nu, theta, n, nO=3.05):
    return 2 * np.pi * nu * (nO - n) * theta / c

def compute_effective_index(angleIncidence, chi, nE=3.38, nO=3.05):
    sin_inc_sq = jnp.sin(angleIncidence)**2
    cos_chi_sq = jnp.cos(chi)**2
    return nE * jnp.sqrt(1 + (nE**-2 - nO**-2) * sin_inc_sq * cos_chi_sq)

def HWP(nu, theta, angleIncidence, chi, nE=3.38, nO=3.05):
    n = compute_effective_index(angleIncidence, chi, nE, nO)
    d = get_delta(nu, theta, n, nO)
    cos_d = jnp.cos(d)
    sin_d = jnp.sin(d)
    
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, cos_d, -sin_d],
        [0.0, 0.0, sin_d, cos_d]
    ])

def rotation_matrix_mueller(angle):
    """4x4 Mueller matrix for rotation"""
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_2a, -sin_2a, 0.0],
        [0.0, sin_2a, cos_2a, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])  # REMOVED: dtype=np.float64


@diagonal
class RealisticHWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate. params = freq, thickness, alpha_2, epsilon, phi"""
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float | None
    angleIncidence: float | None
    epsilon: float | None
    phi: float | None
    
    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float | None,
        angleIncidence:float,
        epsilon:float,
        phi:float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)

        hwp = cls(in_structure, frequency,angleIncidence, epsilon,phi)

        if angles is None:
            return hwp
        angles = angles
        rot = QURotationOperator(angles+phi, in_structure)
        rotated_hwp: AbstractLinearOperator = rot.T @ hwp @ rot
        return rotated_hwp

    def mv(self, x: StokesPyTreeType) -> Stokes:
     
        thickness = thicknesses[3]
        alpha_2 = 54.0*deg # angles[4]
    
   
        HWP1 = HWP(self.frequency*GHz, thickness, self.angleIncidence*deg, chi=0.)
        HWP2_base = HWP(self.frequency*GHz, thickness, self.angleIncidence*deg, chi=alpha_2)
        
        # Full Mueller matrix product
        Mueller_full    = HWP1 @ rotation_matrix_mueller(alpha_2).T @ HWP2_base @ rotation_matrix_mueller(alpha_2) @ HWP1
        Mueller_      = Mueller_full.at[2, :].multiply(-1)  # Q -> -Q for second row
        Mueller_      = Mueller_.at[:, 2].multiply(-1)  # U -> -U for second column
        Mueller_      = Mueller_[:-1, :-1]

        i = Mueller_[0,0] * x.i + Mueller_[0,1] * x.q + Mueller_[0,2] * x.u + Mueller_[0,3] * x.v
        q = Mueller_[1,0] * x.i + Mueller_[1,1] * x.q + Mueller_[1,2] * x.u + Mueller_[1,3] * x.v
        u = Mueller_[2,0] * x.i + Mueller_[2,1] * x.q + Mueller_[2,2] * x.u + Mueller_[2,3] * x.v
        v = Mueller_[3,0] * x.i + Mueller_[3,1] * x.q + Mueller_[3,2] * x.u + Mueller_[3,3] * x.v

        print(Mueller_[3,1], '\n', Mueller_)
            # if isinstance(x, StokesQUPyTree):
            #     return StokesQUPyTree(q, u)
        if isinstance(x, StokesIQU):
            return StokesIQU(i,self.epsilon*q,self.epsilon*u)
        if isinstance(x, StokesIQUV):
            return StokesIQUV(i,q,u,v)
        raise NotImplementedError
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    

@diagonal
class WPOperator(AbstractLinearOperator):
    """Operator for a wave plate."""

    phi: Float[Array, '...']
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        phi: Float[Array, '...'] | None = None,
        angles : Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        wp = cls(phi, in_structure)
        if angles is None:
            return wp
        rot = QURotationOperator(angles, in_structure)
        rotated_wp: AbstractLinearOperator = rot.T @ wp @ rot
        return rotated_wp

    def mv(self, x: StokesPyTreeType) -> Stokes:
        if isinstance(x, StokesI):
            return x
        if isinstance(x, StokesQU):
            return StokesQU(x.q, x.u)
        
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        
        if isinstance(x, StokesIQU):
            return StokesIQU(x.i, x.q, x.u * cos_phi)
        
        u = x.u * cos_phi - x.v * sin_phi
        v = x.u * sin_phi + x.v * cos_phi
        
        if isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, x.q, u, v)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class QURotationHWPRule(AbstractBinaryRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)`."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator):
            return [right, QURotationTransposeOperator(left)]
        assert isinstance(left, QURotationTransposeOperator)
        return [right, left.operator]
