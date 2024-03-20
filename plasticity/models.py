import numpy as np
import pandas as pd
from uniaxial import KinematicUniaxialMaterialModel, AbstractUniaxialMaterialModel
from typing import Optional
from typing import Optional, Literal, Dict, Any
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from scipy import interpolate
import seaborn as sns

sns.set_theme()

"""
Library of placticity models for hardenning modelling
"""

# Kinematic models

class DummyModel(KinematicUniaxialMaterialModel):
    """
    Dummy model. Do nothing. Just Be.
    
    Args:
        
        E: Young modulus
        mu: Poisson ratio
        dt: time step in experiment
        constants: set of initial constants for model
    """
    def __init__(self, E, mu, dt, upper_yield_strength, downer_yield_strength, constants):
        super().__init__(E, mu, dt, upper_yield_strength, downer_yield_strength, constants)
        
    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        return 0
    
class KadaschevichKinematicModel(KinematicUniaxialMaterialModel):
    """
    Kadashevich linear kinematic model.
    
    Args:
        
        E: Young modulus
        mu: Poisson ratio
        dt: time step in experiment
        constants: set of initial constants for model
    """
    def __init__(self, E, mu, dt,  upper_yield_strength, downer_yield_strength, constants):
        super().__init__(E, mu, dt,  upper_yield_strength, downer_yield_strength, constants)

    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        return 2/3*self.constants['g']*strain_rate
    
class ArmstrongKinematicModel(KinematicUniaxialMaterialModel):
    """
    Armstrong kinematic model.
    
    Args:
        
        E: Young modulus
        mu: Poisson ratio
        dt: time step in experiment
        constants: set of initial constants for model
    """
    def __init__(self, E, mu, dt,  upper_yield_strength, downer_yield_strength, constants):
        super().__init__(E, mu, dt,  upper_yield_strength, downer_yield_strength, constants)

    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        intensity = np.sqrt(2/3)*strain_rate
        d_alpha = 2/3*self.constants['g']*strain_rate + self.constants['ga']*alpha*intensity
        return d_alpha
    
class BondarKinematicModel(KinematicUniaxialMaterialModel):
    """
    Bondar kinematic model.
    
    Args:
        
        E: Young modulus
        mu: Poisson ratio
        dt: time step in experiment
        constants: set of initial constants for model
    """
    def __init__(self, E, mu, dt,  upper_yield_strength, downer_yield_strength, constants):
        super().__init__(E, mu, dt,  upper_yield_strength, downer_yield_strength, constants)

    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        intensity = np.sqrt(2/3)*strain_rate
        g = self.constants['Ea'] + self.constants['Betta']*self.constants['Sigma']
        ge = self.constants['Betta']*self.constants['Ea']
        ga = -self.constants['Betta']
        d_alpha = 2/3*g*strain_rate + (2/3*ge*strain + ga*alpha)*intensity
        return d_alpha
    

class ChabocheKinematicModel(KinematicUniaxialMaterialModel):
    """
    Bondar kinematic model.
    
    Args:
        
        E: Young modulus
        mu: Poisson ratio
        dt: time step in experiment
        constants: set of initial constants for model
    """
    def __init__(self, E, mu, dt,  upper_yield_strength, downer_yield_strength, constants):
        super().__init__(E, mu, dt,  upper_yield_strength, downer_yield_strength, constants)

    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        d_alphas = []
        intensity = np.sqrt(2/3)*strain_rate
        for i in range(1, int(self.constants['n_alphas']) + 1):
            d_alpha_i = 2/3*self.constants[f'g_{i}']*strain_rate + self.constants[f'ga_{i}']*self.constants[f'a_{i}']*intensity
            self.constants[f'a_{i}'] += d_alpha_i
            d_alphas.append(d_alpha_i)

        d_alpha = sum(d_alphas)
        return d_alpha
    

# Isotropic Model
class IsotropicModel(AbstractUniaxialMaterialModel):
    def __init__(self, E, mu, dt, constants: dict = {}):
        super().__init__(E, mu, dt, constants)

    def predict(
                self, 
                strain: np.ndarray=None, 
                stress: np.ndarray=None, 
                initial_value: float=0,
                yield_strength: Optional[float]=None,
                input_mode: Literal['strain', 'stress'] = 'strain'):
        R = yield_strength
        chi = 0
        chi_log = []
        if input_mode == 'strain':
            preds = []
            plastic = [0]
            elastic = [0]
            current_stress = initial_value

            for current_strain in strain:
                elastic_strain = current_stress/self.E
                plastic_strain = current_strain - elastic_strain
                elastic.append(elastic_strain)
                plastic.append(plastic_strain)
                d_plastic = plastic[-1] - plastic[-2]
                
                suggestion = self.E*current_strain
                if suggestion>R:
                    chi += np.sqrt(2/3*d_plastic**2)
                    try:
                        new_R = self.expansion_funtion(chi)
                        R = new_R
                    except ValueError:
                        pass
                        #print('Warning: The boundary of Interpolation was reached. The following interpolation is constant.')
                    current_stress = R

                else:
                    current_stress = suggestion
                preds.append(current_stress)
                chi_log.append(chi)
        self.pred_chi = chi_log
        return preds

    def fit(self, 
            strain: np.array,
            stress: np.array,
            yield_strength: Optional[float]=None,
            input_mode: Literal['strain', 'stress']='strain',
            plot:bool=False):
        
        if input_mode == 'strain':
            plastic_strain, elastic_strain = self.plastic_elastic_decomposition(strain=strain, stress=stress)
            plastic_strain_rate = self.get_strain_rate(plastic_strain)*self.dt
            chi = np.cumsum(np.sqrt(2/3*plastic_strain_rate**2))
            R = np.copy(stress)
            R[stress<yield_strength] = yield_strength
            first_out = max((R == yield_strength).argmin() - 1, 0)
            R = R[first_out:]
            chi = chi[first_out:] 
            #poly = lagrange(chi, R)
            #self.expansion_funtion = lambda chi_: Polynomial(poly.coef[::-1])(chi_)
            self.f = interpolate.interp1d(np.append(chi, [0]), np.append(R, [yield_strength]))
            self.expansion_funtion = lambda chi_: self.f(chi_)

            self.train_chi = chi
            if plot:
                train_preds = self.expansion_funtion(chi)
                plt.plot(chi, train_preds, label='R from Chi')
                plt.xlabel('Chi')
                plt.ylabel('R')
                plt.legend()
                plt.show()


