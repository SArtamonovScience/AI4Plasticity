import numpy as np
import typing
from typing import Optional, Literal, Dict
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from utils import logger

REVERSE_MODE_TABLE = {0: 2, 1: 2, 2: 0, 3: 0}
class AbstractUniaxialMaterialModel(object):
    def __init__(self, E, mu, dt, upper_yield_strength, downer_yield_strength, constants: dict={}):
        self.E = E
        self.mu = mu
        self.dt = dt
        self.upper_yield_strength = upper_yield_strength
        self.downer_yield_strength = downer_yield_strength
        self.constants = constants
        self.G = E/(2*(mu+1))
        
    def get_c_names(self):
        return list(self.constants.keys())
        
    def get_c_vals(self):
        return np.array(list(self.constants.values()))
        
    def fit(self, strain: np.array, stress: np.array):
        pass
    
    def predict(self, 
                strain: Optional[np.array]=None, 
                stress: Optional[np.array]=None, 
                input_mode: Literal['strain', 'stress']='strain'):
        pass

        
    
    def get_yield_strengt(self, strain: np.array, stress: np.array, tolerance: float=1e-2) -> float:
        """
            Function for determining the ultimate elastic stress
            for uniaxial experiments.
        
            Args:

                strain: sequence of strain measurements, %.
                stress: sequence of stress measurements, Pa.
                tolerance: the maximum value of the relative error of the experiment.
                
            Returns:

                Yield Strength, Pa
        """
        elastic_strain = stress/self.E
        relative_deviation = abs(strain - elastic_strain)/strain
        is_elastic = relative_deviation<tolerance
        yield_strength = stress[is_elastic][0]
        return yield_strength
    
    
    def plastic_elastic_decomposition(self, strain: np.array, stress: np.array, tolerance: float=1e-2) -> float:
        """
            In all of placticity theories the hypothesis about plastic-elastic decomposition of
            strain is accepted:
                                    strain = strain_plastic + strain_elastic
            This funtion provides such decomposition.
            
            Args:
                strain: sequence of strain measurements, %.
                stress: sequence of stress measurements, Pa.
                tolerance: the maximum value of the relative error of the experiment.
                
            Returns:
                plastic and elastic parts of strain.
        """
        
        ### Переделать для разгружения с некоторой пластичности!! (Вроде все нормально)
        strain_ = np.copy(strain)
        elastic_strain = stress/self.E
        plastic_strain = strain_ - elastic_strain
        return plastic_strain, elastic_strain
    
    
    def get_strain_rate(self, strain: np.array):
        """
            Returns strain rate from strain curve.
            
            Args:
                strain: sequence of strain measurements, %.
                
            Returns:
            
                Strain rate.
        """
        
        strain_shifted = np.roll(strain, 1)
        strain_shifted[0] = 0
        strain_rate = (strain - strain_shifted)/self.dt
        
        return strain_rate
    
    def set_mean_plastic_strain(self, value=None, plastic_strain=None):
        assert value is not None or plastic_strain is not None, 'One of value or plastic_strain must be specified'
        if value is not None:
            self.mean_plastic_strain = value
        else:
            self.mean_plastic_strain = np.mean(plastic_strain)

    def set_mean_plastic_strain_rate(self, value=None, plastic_strain=None, plastic_strain_rate=None):
        assert value is not None or plastic_strain is not None or plastic_strain_rate is not None, 'One of value or plastic_strain or plastic_strain_rate must be specified'
        if value is not None:
            self.mean_plastic_strain_rate = value
        elif plastic_strain is not None:
            plastic_strain_rate = self.get_strain_rate(plastic_strain)
            self.mean_plastic_strain_rate = np.mean(abs(plastic_strain_rate))
        else:
            self.mean_plastic_strain_rate = np.mean(abs(plastic_strain_rate))
    
    
class KinematicUniaxialMaterialModel(AbstractUniaxialMaterialModel):
    # TODO: (1) адаптировать под разные варианты циклических нагружений, а не только под последовательный цикл в одну сторону
    #       (2) добавить нормировку и обезразмеривание
    #       (3) добавить возможность использования нескольких альфа
    #       (4) добавить возможность выбора оптимизатора
    #       (5) возможность учиться на нескольких экспериментах
    def __init__(self, E, mu, dt,  upper_yield_strength, downer_yield_strength, constants: dict):
        super().__init__(E, mu, dt, upper_yield_strength, downer_yield_strength)
        self.backstress_params = None
        self.constants = constants
        self.logger = logger(['plastic', 'alpha', 'd_alpha', 'plastic_strain_rate', 'strain_rate', 'strain', 'stress', 'colors'])
    
    def get_c_names(self):
        return list(self.constants.keys())
        
    def get_c_vals(self):
        return np.array(list(self.constants.values()))
    
    
    def fit(self, 
                strain: np.array,
                stress: np.array,
                initial_value: float=0,
                upper_yield_strength: Optional[float]=None,
                downer_yield_strength: Optional[float]=None,
                input_mode: Literal['strain', 'stress']='strain',
                tolerance: float=1e-3):
        
        if input_mode == 'strain':
            
            plastic_strain, _ = self.plastic_elastic_decomposition(strain=strain, stress=stress, tolerance=tolerance)
            self.set_mean_plastic_strain(plastic_strain=plastic_strain)
            self.set_mean_plastic_strain_rate(plastic_strain=plastic_strain)

            def objective(x):
                self.constants = dict(zip(self.get_c_names(), x))
                preds = self.predict(strain=strain, 
                                     initial_value=initial_value,
                                     input_mode='strain',
                                     tolerance=tolerance)
                #err = np.mean((preds - stress)**2)
                err = preds - stress
                return err

            #res = minimize(objective, self.get_c_vals(), method='powell',
            #       options={'xatol': 1e-15, 'disp': True})
            res = least_squares(objective, self.get_c_vals())
            self.constants = dict(zip(self.get_c_names(), res.x))
            self.res = res

    def _predict(self, 
                strain: Optional[np.array]=None,
                stress: Optional[np.array]=None,
                initial_value: float=0,
                yield_strength: Optional[float]=None,
                input_mode: Literal['strain', 'stress']='strain',
                tolerance: float=1e-1):
        self.logger.reload()
        if input_mode == 'strain':
            current_stress = initial_value
            plastic = 0
            elastic = 0
            self.logger.log('plastic', 0)
            strain_rate = self.get_strain_rate(strain=strain)
            in_plastic = False
            alpha = 0
            
            for i, current_strain in enumerate(strain):
                loading = strain_rate[i]>0
                elastic_strain = current_stress/self.E
                plastic_strain = current_strain - elastic_strain
                d_plastic = plastic_strain - self.logger.get_last('plastic')
                plastic += d_plastic
                plastic_strain_rate = (d_plastic)/self.dt

                if plastic/current_strain > tolerance:
                    in_plastic = True

                if loading:
                    if in_plastic:
                        d_alpha = self._d_alpha_(strain=plastic_strain, 
                                 strain_rate=plastic_strain_rate, 
                                 alpha=alpha)
                        alpha += d_alpha
                        current_stress = alpha + yield_strength
                        self.logger.log('colors', 'r')

                    else:
                        current_stress = current_stress + strain_rate[i]*self.dt*self.E
                        d_alpha = 0
                        self.logger.log('colors', 'm')
                else:
                    if in_plastic:
                        d_alpha = self._d_alpha_(strain=plastic_strain, 
                                 strain_rate=plastic_strain_rate, 
                                 alpha=alpha)
                        alpha += d_alpha
                        current_stress = alpha - yield_strength
                        self.logger.log('colors', 'b')
                    else:
                        current_stress = current_stress + strain_rate[i]*self.dt*self.E
                        d_alpha = 0
                        self.logger.log('colors', 'g')



                self.logger.log('plastic', plastic_strain)
                self.logger.log('d_alpha', d_alpha)
                self.logger.log('alpha', alpha)
                self.logger.log('stress', current_stress)
            return self.logger.get('stress')
            
        
    def mode_1_deformation(self, alpha, strain, strain_rate, strain_plastic, yield_strengt):
        """
            Case of loading in elastic domain
            Args:
                alpha: current state of backstress
                strain:
                yield_strengt
        """
        #sigma_pred = current_stress + self.E*(strain_rate*self.dt)
        #sigma_pred = self.E*strain
        sigma_pred = self.E*(strain - strain_plastic)
        change_mode_flag = False
        d_alpha = 0
        if sigma_pred > alpha*self.G  + yield_strengt*1.01:
            d_strain = strain_rate*self.dt
            d_elastic_strain = ((alpha*self.G + yield_strengt) - (strain - d_strain)*self.E)/self.E
            d_plastic_strain = d_strain - d_elastic_strain
            plastic_strain = d_plastic_strain
            plastic_strain_rate = plastic_strain/self.dt
            sigma_pred, _, d_alpha = self.mode_2_deformation(alpha, plastic_strain, plastic_strain_rate, yield_strengt)
            change_mode_flag = True
        return sigma_pred, change_mode_flag, d_alpha
    
    def mode_2_deformation(self, alpha, plastic_strain, plastic_strain_rate, yield_strengt, mode=1):
        """
            Case of loading and unloading in plastic domain
        """
        d_alpha = self._d_alpha_(strain=plastic_strain/self.mean_plastic_strain, 
                                 strain_rate=plastic_strain_rate/self.mean_plastic_strain_rate, 
                                 alpha=alpha)
        if mode==1:
            sigma_pred = (alpha + d_alpha)*self.G + yield_strengt
        else:
            sigma_pred = (alpha + d_alpha)*self.G - yield_strengt
        return sigma_pred, False, d_alpha
        
    def mode_3_deformation(self, alpha, strain, strain_rate, strain_plastic, yield_strengt):
        """
            Case of unloading in elastic domain
        """
        #sigma_pred = current_stress + self.E*(strain_rate*self.dt)
        #sigma_pred = self.E*strain
        sigma_pred = self.E*(strain - strain_plastic)
        change_mode_flag = False
        d_alpha = 0
        
        if sigma_pred < alpha*self.G  - yield_strengt*1.01:
            d_strain = strain_rate*self.dt
            d_elastic_strain = ((alpha*self.G - yield_strengt) - (strain - d_strain)*self.E)/self.E
            d_plastic_strain = d_strain - d_elastic_strain
            plastic_strain = d_plastic_strain
            plastic_strain_rate = plastic_strain/self.dt
            sigma_pred, _, d_alpha = self.mode_2_deformation(alpha, plastic_strain, plastic_strain_rate, yield_strengt, mode=3)
            change_mode_flag = True
        return sigma_pred, change_mode_flag, d_alpha
    
    def step(self, 
             current_strain,
             last_stress,
             cumulative_plastic_strain,
             alpha
             ):
        suggestion = (current_strain - cumulative_plastic_strain)*self.E
        if alpha*self.G - self.downer_yield_stress <= suggestion <= alpha*self.G + self.upper_yield_stress:
            d_alpha = 0
            prediction = suggestion
            d_plastic_strain = 0
        else:
            elastic_strain = last_stress/self.E
            plastic_strain = current_strain - elastic_strain
            d_plastic_strain = plastic_strain - cumulative_plastic_strain
            plastic_strain_rate = d_plastic_strain/self.dt
            d_alpha = self._d_alpha_(strain=plastic_strain/self.mean_plastic_strain, 
                                     strain_rate=plastic_strain_rate/self.mean_plastic_strain_rate, 
                                     alpha=alpha)
            alpha = alpha + d_alpha
            prediction = alpha*self.G + self.upper_yield_stress if d_alpha > 0 else alpha*self.G - self.downer_yield_stress
        return prediction, d_alpha, d_plastic_strain

       
    def predict__(self, 
                strain: Optional[np.array]=None,
                stress: Optional[np.array]=None,
                initial_value: float=0,
                input_mode: Literal['strain', 'stress']='strain',
                tolerance: float=1e-2):

        if input_mode == 'strain':
            sigma_current = initial_value
            predictions = []
            alpha = 0
            cumulative_plastic_strain = 0
            for i, current_strain in enumerate(strain):
                prediction, d_alpha, d_plastic_strain = self.step(current_strain=current_strain,
                                                                  last_stress=sigma_current,
                                                                  cumulative_plastic_strain=cumulative_plastic_strain,
                                                                  alpha=alpha)
                predictions.append(prediction)
                alpha += d_alpha
                cumulative_plastic_strain += d_plastic_strain
            return predictions



        
    def predict(self, 
                strain: Optional[np.array]=None,
                stress: Optional[np.array]=None,
                initial_value: float=0,
                upper_yield_strength: Optional[float]=None,
                downer_yield_strength: Optional[float]=None,
                input_mode: Literal['strain', 'stress']='strain',
                tolerance: float=1e-2):
        """
            Function to make predictions about strain or stress in some experiment.
            In Kinematic Hardening size of yield surface is considered to be constant.
            In the case of uniaxial experiments we need just to move "window of elasticity" with
            constant width for a distance equal to current value of backstress.
            
            Args:
                
                strain: 
                    sequence of strain measurements, %.
                stress: 
                    sequence of stress measurements, Pa.
                initial_value:
                    initial value of strain or stress.
                yield_strength: 
                    yield strength of the material (work piece), Pa. If None, will be used from fit.
                input_mode: 
                    what the input data is.
                tolerance: 
                    the maximum value of the relative error of the experiment.
                
            Returns:
                
                Predictions of the model.
        """
        upper_yield_strength = upper_yield_strength if upper_yield_strength else self.upper_yield_strength
        downer_yield_strength = downer_yield_strength if downer_yield_strength else self.downer_yield_strength

        yield_strength = upper_yield_strength
        yield_strength = yield_strength if yield_strength else self.yield_strength
        if input_mode == 'strain':
            sigma_current = initial_value
            strain_rate = self.get_strain_rate(strain)
            loading_flag = (strain_rate>=0).astype(int)
            elastic = np.array([])
            plastic = np.array([])
            alpha = 0
            stress_predicted = []
            p_rate = []
            d_alphas = []
            alphas = []
            modes = []
            current_mode = 0

            cumulative_plastic = 0
            for i, current_strain in enumerate(strain):
                if i == 0:
                    is_reverse = False
                else:
                    is_reverse = (loading_flag[i] - loading_flag[i-1]) != 0
                if is_reverse:
                    current_mode = REVERSE_MODE_TABLE[current_mode]
                    
                if current_mode == 0:
                    sigma_pred, change_mode_flag, d_alpha = self.mode_1_deformation(alpha, current_strain, strain_rate[i], cumulative_plastic, yield_strength)
                    elastic = np.append([current_strain], elastic)
                    plastic = np.append([cumulative_plastic], plastic)
                    p_rate.append(0)
                    if change_mode_flag:
                        current_mode = 1
                
                elif current_mode == 1 or current_mode == 3:
                    elastic_strain = sigma_current/self.E
                    plastic_strain = current_strain - elastic_strain                    
                    elastic = np.append([elastic_strain], elastic)
                    plastic = np.append([plastic_strain], plastic)
                    d_plastic = plastic[0] - plastic[1]
                    cumulative_plastic += d_plastic
                    plastic_strain_rate = (d_plastic)/self.dt
                    p_rate.append(plastic_strain_rate)         
                    if current_mode == 1:           
                        sigma_pred, change_mode_flag, d_alpha = self.mode_2_deformation(alpha, plastic_strain, plastic_strain_rate, yield_strength, mode=1)
                    else:
                        sigma_pred, change_mode_flag, d_alpha = self.mode_2_deformation(alpha, plastic_strain, plastic_strain_rate, downer_yield_strength, mode=3)
                
                elif current_mode == 2:
                    sigma_pred, change_mode_flag, d_alpha = self.mode_3_deformation(alpha, current_strain, strain_rate[i], cumulative_plastic, downer_yield_strength)
                    elastic = np.append([current_strain], elastic)
                    plastic = np.append([cumulative_plastic], plastic)
                    p_rate.append(0)
                    if change_mode_flag:
                        current_mode = 3
                else:
                    pass
                sigma_current = sigma_pred
                alpha+=d_alpha
                d_alphas.append(d_alpha)
                alphas.append(alpha)
                stress_predicted.append(sigma_current)
                modes.append(current_mode)
            
            self.elastic = elastic[::-1]
            self.plastic = plastic[::-1]
            self.d_alphas = d_alphas
            self.alphas = alphas
            self.p_rate = p_rate
            self.modes = modes
            return np.array(stress_predicted)

    
    def _d_alpha_(self, 
                    strain: Optional[np.array]=None, 
                    strain_rate: Optional[np.array]=None, 
                    stress: Optional[np.array]=None,
                    alpha: Optional[np.array]=None,
                    **kwargs):
        """
        Function to model d_alpha/G (normalized d_alpha)

        :param strain: _description_, defaults to None
        :type strain: Optional[np.array], optional
        :param strain_rate: _description_, defaults to None
        :type strain_rate: Optional[np.array], optional
        :param stress: _description_, defaults to None
        :type stress: Optional[np.array], optional
        :param alpha: _description_, defaults to None
        :type alpha: Optional[np.array], optional
        """
        pass
    
#class CombinedUniaxialModel():
            

