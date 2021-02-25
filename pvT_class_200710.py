# -*- coding: utf-8 -*-
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
====         ===  ==       ==  ==           |      M ontan
||\\        //||  ||       ||  ||           |      U niversitaet
|| \\      // ||  ||       ||  ||           |      L eoben
||  \\    //  ||  ||       ||  ||           |
||   \\  //   ||  ||       ||  ||           |      Institute for 
||    \\//    ||  ||       ||  ||           |      polymer processing
||            ||  ||       ||  ||           |
||            ||   =========    ==========  |      Author:    Sykwalker
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This class is for evaluation of a pvT curve, to detected the transition 
temperatures as a function of the pressure. 
"""

"""
Importing of all necessary packages of python
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, differential_evolution, NonlinearConstraint, least_squares, minimize


import os
import time
import glob


class pvT_analysis:
    
    def __init__(self, fname, polymer_type, **kwargs):
        self.readtxt(fname, **kwargs)
        self.__polymer_type = polymer_type
        """
        initialize the call internal varables
        """
        
        self.__T_v_p_melt = np.empty((0,3))
        self.__T_v_p_solid = np.empty((0,3))
        self.__T_p_at_trans = np.empty(0)
        
        self.__coeffs = {c : None for c in ['b5', 'b6', 'b7', 'b8', 'b9',
                                             'b1m', 'b2m', 'b3m', 'b4m', 
                                             'b1s', 'b2s', 'b3s', 'b4s']}
        self.__coeffs_old = {c : None for c in ['b5', 'b6', 'b7', 'b8', 'b9',
                                             'b1m', 'b2m', 'b3m', 'b4m', 
                                             'b1s', 'b2s', 'b3s', 'b4s']}
        
        self.__dic_bounds_AMI ={'b5':(0,1000), 'b6':(0,5e-5), 'b7':(0,0.1), 'b8':(0,50), 'b9':(0,1e-5),
                                'b1m':(2e-6,0.2), 'b2m':(1e-9,0.0001), 'b3m':(500000,5e10), 'b4m':(1e-7,1),
                                'b1s':(2e-6,0.2), 'b2s':(1e-9,0.0001), 'b3s':(500000,5e10), 'b4s':(1e-7,1)}
        
        self.__color = ['black', 'red', 'blue', 'green', 'orange', 'grey', 'yellow']
        pass
    
    def readtxt(self, fname, **kwargs):
        """
        Read in the pvT.txt file provided by our machines.
        """

        options = {
            'dtype' : np.str,
            'delimiter'  : ';',
            'skiprows' : 2
            }
    
        options.update(kwargs)
        
        a = np.loadtxt(fname, dtype = options['dtype'], 
                       delimiter=options['delimiter'], 
                       skiprows=options['skiprows'])
        
        d = np.empty((0,a.shape[1]-2))
        
        for row in a:                
            d = np.vstack((d, np.asarray([float(s.replace(',', '.')) for s in row[1:-1]])))
    
        f = open(fname, 'r')
        header_row = f.readlines()[1].split(';')[1:-1]
        header =  [header_row[0][:1]]
        for h in header_row[1:]:
            header.append(h[h.find(r'(')+1:h.find(r')')].replace(',', '.'))
            
        f.close()
        
        self.__data = pd.DataFrame(d, columns=header)
        #print(self.__data)
        return self.__data
    
    
    """
    Different equations for fitting of the pvT Data. 
    Three functions are present for fitting the transition Temperature as a function of pressure, 
    the specivic volume in melt state and the specifiv volume in soid state. 
    
    In the end the Data are used 
    
    """
    def func_Ttrans(self, p, b5, b6):
        """
        Equation of the Transition temperature as definded in the Tait equations.
        """
        return(b5+b6*p) #=Ttrans(p) #in Si
        
        
    def func_v_melt_state(self, p,T,b1m, b2m, b3m, b4m,b5):
        """
        Equation of the melt state as definded in the Tait equations.
        """
        C=0.0894
        
        v0=b1m+b2m*(T-b5)
        
        B=b3m*np.exp(-b4m*(T-b5))
        
        v_spez =v0*(1-C*np.log(1+p/B))
        
        return v_spez #v(T,p) #in Si
    

    def func_v_solid_state(self, p,T,b5,b1s,b2s,b3s,b4s,b7,b8,b9):
        """
        Equation of the solid state as definded in the Tait equations.
        """
        C=0.0894
        v0=b1s+b2s*(T-b5)
        
        try: B=b3s*np.exp(-b4s*(T-b5))
        except OverflowError: B=float('inf')
        
        try: vt =b7*np.exp((b8*(T-b5))-(b9*p))
        except OverflowError: vt=float('inf')
        
        v_spez =(v0*(1.-C*np.log(1.+p/B))+vt)
        return v_spez #v(T,p) #in Si
    
    
    def func_pvT_master(self, p, TT):
        v = np.empty(0)
        for T in TT:
            
            if T < self.func_Ttrans(p, self.__coeffs['b5'],self.__coeffs['b6']):
                v_spez = self.func_v_solid_state(p, T, self.__coeffs['b5'], self.__coeffs['b1s'], 
                                        self.__coeffs['b2s'], self.__coeffs['b3s'], 
                                        self.__coeffs['b4s'], self.__coeffs['b7'], 
                                        self.__coeffs['b8'], self.__coeffs['b9'])
            else:
                v_spez = self.func_v_melt_state(p, T, self.__coeffs['b1m'], self.__coeffs['b2m'], 
                                            self.__coeffs['b3m'], self.__coeffs['b4m'],
                                            self.__coeffs['b5'])
            v = np.append(v, v_spez)    
        return v
            
    """
    We have to split the date into solid state and melt state, thus we find the 
    maximum value for amorphous materials and the minimum value for semi-crystaline values
    """  
    
    def find_Ttrans(self, guess = 100, **kwargs):
        options = {
                'maxInt' : 100,
                'dT' : 10,
                'convergence' : 1e-15,
                'relaxation' : 1
                }
        options.update(kwargs)
        
        T_trans = []
        
        T = self.__data.loc[:]['T'].to_numpy()
        v = self.__data.drop(columns='T').to_numpy()
        p = self.__data.columns[1:].to_numpy()
        
        for p_v, v_v in zip(p,v.T):
            
            dv = np.gradient(v_v)
            dv = savgol_filter(dv, 17,3)
            ddv = np.gradient(dv)
            ddv = savgol_filter(ddv, 17,3)

            s = (np.abs(T-(guess-40))).argmin()
            e = (np.abs(T-(guess+40))).argmin()
    
            T1 = T[e:s]
            dv1 = dv[e:s]
            ddv1 = ddv[e:s]

            if self.__polymer_type == 'sc':
                Ttrans = T1[(ddv1).argmin()]
            else:
                Ttrans = T1[(ddv1).argmax()]
            #print('Ttrans = {}'.format(Ttrans))
            for i in range(options['maxInt']):
                
                s = (np.abs(T-(Ttrans-options['dT']))).argmin()
                m = (np.abs(T-(Ttrans))).argmin()
                e = (np.abs(T-(Ttrans+options['dT']))).argmin()
                
                tm =[T[e:m], T[m:s]]
                vm = [v_v[e:m], v_v[m:s]]
                
                cc = [np.polyfit(x_dat, y_dat, 1) for x_dat, y_dat in zip(tm, vm)]
                Ttrans_new = (cc[1][1]-cc[0][1])/(cc[0][0]-cc[1][0])
                
                if options['convergence'] > (1-Ttrans_new/Ttrans):
                    print(r'$T_{trans}$ = '+str(Ttrans)+ ' after {} Iterations'.format(i+1))
                    break
                else:
                    Ttrans = Ttrans_new * options['relaxation']
                    
            T_trans.append([Ttrans+273.15, float(p_v)*1e5])       
                
            
        self.__T_p_at_trans = np.asarray(T_trans)
        
        
    def split_data(self, p, v, T):
        
        Ttrans = self.__T_p_at_trans[:,0]
        
        for p_v, v_v, tt in zip(p,v.T, Ttrans):
            
            
            i = (np.abs(T-(tt-273.15)).argmin())
            #print(i)
            m = np.vstack((T[:i]+273.15, v_v[:i]*1e-3, np.full(len(T[:i]), (float(p_v)*1e5)))).T
            s = np.vstack((T[i:]+273.15, v_v[i:]*1e-3, np.full(len(T[i:]), (float(p_v)*1e5)))).T
            
            
            self.__T_v_p_solid = np.vstack((self.__T_v_p_solid, s))
            self.__T_v_p_melt = np.vstack((self.__T_v_p_melt, m))
        
        pass


    """
    Now we start to fit the diffferent equations with curve fit for the transition temperature 
    and with differential evolution for the other equations 
    """    
    def fit_T_at_trans(self):
        """
        Fit Transition Temperature
        """
        coeffs = ['b5', 'b6']
        
        b5_b6 = np.asarray([self.__dic_bounds_AMI.get(c) for c in coeffs])
        
        bounds_AMI_b5b6 = np.asarray(b5_b6).T.tolist()
        
        bounds_AMI_b5b6 = tuple([tuple(row)for row in bounds_AMI_b5b6])
        
        ini_coef_b5b6= tuple([np.mean(row) for row in b5_b6]) #initals values for b5 and b6
        
        T_trans = self.__T_p_at_trans[:,0]
        p_T_trans = self.__T_p_at_trans[:,1]
        
        popt, pcov = curve_fit(self.func_Ttrans, p_T_trans, T_trans,
                               p0=ini_coef_b5b6, bounds=bounds_AMI_b5b6)
        
        b5,b6=popt
        for c, v, in zip(coeffs, popt):
            self.__coeffs[c] = v
        
        #print(f' b5: {b5:.2e} \n b6: {b6:.2e}')
        return popt, pcov
        
        
    def fit_melt_state(self):
        self.__iter = 0
        coeffs = ['b1m', 'b2m', 'b3m', 'b4m']
        bounds_AMI_melt_state = tuple([self.__dic_bounds_AMI.get(c) 
                                        for c in coeffs])
        
        #Equations:
        def mini_Tait_melt_state(coef,*arg):
            """
            For minimizing the melt state. Min the sum of square.
            """
            b1m, b2m, b3m, b4m =coef
            T_v_p_melt, b5 = arg

            error=np.zeros(1)
            
            for T,v,p in T_v_p_melt:
                v_calc = self.func_v_melt_state(p,T,b1m, b2m, b3m, b4m, b5)
                error+=(v-v_calc)**2.
            #error=np.sqrt(error/float(len(T_v_p_melt)))

            return error
   
        #Actual fitting:
        popt=differential_evolution(mini_Tait_melt_state,
                            args=(self.__T_v_p_melt,self.__coeffs['b5'],),
                            bounds=bounds_AMI_melt_state, disp = False)

        success=popt.success
        nb_iterations=popt.nit
        message=popt.message
        b1m, b2m, b3m, b4m=popt.x
        
        for c, b in zip(coeffs, popt.x):
            self.__coeffs[c] = b

        
        #print(f' b1m: {b1m:.2e} \n b2m: {b2m:.2e} \n b3m: {b3m:.2e} \n b4m: {b4m:.2e}')
        #print(f'(Nb of iterations performed: {nb_iterations}, Success: {success})')
        #print(f'Message from solver: {message}')
        return popt
       
    
    def fit_solid_state(self, method):
        coeffs = ['b1s', 'b2s', 'b3s', 'b4s', 'b7', 'b8', 'b9']
        bounds_AMI_solid_state = tuple([self.__dic_bounds_AMI.get(c) 
                                        for c in coeffs])
        def constraint_func(coef,*arg):
            b1s, b2s, b3s, b4s, b7, b8, b9, = coef
            b5, b6, b1m, b2m, b3m, b4m = arg
            pressures=(0.,2000*1e5) #same length as the bound tuple ( (0.,0.) and (np.inf,np.inf) )
            l_v_delta=[]
            
            for p in pressures:
                Ttrans_calc=self.func_Ttrans(p,b5,b6)
                v_delta=self.func_v_melt_state(p,Ttrans_calc,b1m, b2m, b3m, b4m,b5) - self.func_v_solid_state(p,Ttrans_calc,b5,b1s,b2s,b3s,b4s,b7,b8,b9)
                l_v_delta+=[float(v_delta)]
            return l_v_delta
        
        
        b5,b6,b1m,b2m,b3m,b4m = (self.__coeffs.get(c) for c in ['b5','b6','b1m','b2m','b3m','b4m'])
        
        constraint= lambda coef: constraint_func(coef, b5,b6,b1m,b2m,b3m,b4m)
        nlc=NonlinearConstraint(constraint,(0.,0.),(np.inf,np.inf),keep_feasible=True)    
    
        #Equations:
        def mini_Tait_solid_state(coef,*arg):
            """
            For minimizing the melt state. Min the sum of square.
            """
            b1s, b2s, b3s, b4s, b7, b8, b9, =coef
            T_v_p_solid,b5= arg
            error=np.zeros(1)
            for T,v,p in T_v_p_solid:
                v_calc=self.func_v_solid_state(p,T,b5,b1s,b2s,b3s,b4s,b7,b8,b9)
                error+=(v-v_calc)**2.
            #error=np.sqrt(error/float(len(T_v_p_solid[:,0])))  #It really does not like that line...
            return error
            
        #Actual fitting:
        popt=differential_evolution(mini_Tait_solid_state,
                                    args=(self.__T_v_p_solid,self.__coeffs['b5'])
                                    ,bounds=bounds_AMI_solid_state, strategy=method, #updating = 'immediate', 
                                    tol=1e-5, disp = False)#, constraints=(nlc)) #I have to go with this tolerance otherwise 
                                                                #I always get an error. If you know why -> pls call me...
    
    
        success=popt.success
        nb_iterations=popt.nit
        message=popt.message
        b1s, b2s, b3s, b4s, b7, b8, b9=popt.x
        
        for c, b in zip(coeffs, popt.x):
            self.__coeffs[c] = b

        #print(f'-------------------------------\nThe method {method} gives: \n')
        #print(f' b1s: {b1s:.2e} \n b2s: {b2s:.2e} \n b3s: {b3s:.2e} \n b4s: {b4s:.2e} \n b7: {b7:.2e} \n b8: {b8:.2e} \n b9: {b9:.2e} \n')
        #print(f'(Nb of iterations performed: {nb_iterations}, Success: {success})')
        #print(f'Message from solver: {message} at {popt.fun}')
        return popt


    def fit_solid_state_amorph(self, method):
        coeffs = ['b1s', 'b2s', 'b3s', 'b4s']
        bounds_AMI_solid_state = tuple([self.__dic_bounds_AMI.get(c) 
                                        for c in coeffs])
        def constraint_func(coef,*arg):
            b1s, b2s, b3s, b4s, = coef
            b5, b6, b1m, b2m, b3m, b4m, b7, b8, b9, = arg
            pressures=(0.,2000*1e5) #same length as the bound tuple ( (0.,0.) and (np.inf,np.inf) )
            l_v_delta=[]
            
            for p in pressures:
                Ttrans_calc=self.func_Ttrans(p,b5,b6)
                v_delta=self.func_v_melt_state(p,Ttrans_calc,b1m, b2m, b3m, b4m,b5) - self.func_v_solid_state(p,Ttrans_calc,b5,b1s,b2s,b3s,b4s,b7,b8,b9)
                l_v_delta+=[float(v_delta)]
            return l_v_delta
        
        
        b5,b6,b1m,b2m,b3m,b4m = (self.__coeffs.get(c) for c in ['b5','b6','b1m','b2m','b3m','b4m'])
        
        constraint= lambda coef: constraint_func(coef, b5,b6,b1m,b2m,b3m,b4m)
        nlc=NonlinearConstraint(constraint,(0.,0.),(np.inf,np.inf),keep_feasible=True)    
    
        #Equations:
        def mini_Tait_solid_state(coef,*arg):
            """
            For minimizing the melt state. Min the sum of square.
            """
            b1s, b2s, b3s, b4s, =coef
            T_v_p_solid,b5, b7, b8, b9,= arg
            error=np.zeros(1)
            for T,v,p in T_v_p_solid:
                v_calc=self.func_v_solid_state(p,T,b5,b1s,b2s,b3s,b4s,b7,b8,b9)
                error+=(v-v_calc)**2.
            #error=np.sqrt(error/float(len(T_v_p_solid[:,0])))  #It really does not like that line...
            return error
            
        #Actual fitting:
        popt=differential_evolution(mini_Tait_solid_state,
                                    args=(self.__T_v_p_solid,self.__coeffs['b5'], 0,0,0)
                                    ,bounds=bounds_AMI_solid_state, strategy=method, updating = 'immediate', 
                                    tol=1e-5, disp = False)#, constraints=(nlc)) #I have to go with this tolerance otherwise 
                                                                #I always get an error. If you know why -> pls call me...
    
    
        success=popt.success
        nb_iterations=popt.nit
        message=popt.message
        b1s, b2s, b3s, b4s =popt.x
        
        for c, b in zip(coeffs, popt.x):
            self.__coeffs[c] = b
            
            
        for c in ['b7', 'b8', 'b9']:
            self.__coeffs[c] = 0
            
        #print(f'-------------------------------\nThe method {method} gives: \n')
        #print(f' b1s: {b1s:.2e} \n b2s: {b2s:.2e} \n b3s: {b3s:.2e} \n b4s: {b4s:.2e} \n')
        #print(f'(Nb of iterations performed: {nb_iterations}, Success: {success})')
        #print(f'Message from solver: {message} at {popt.fun}')
        return popt

    def get_real_Ttrans(self):
        coeffs = ['b5', 'b6']
        
        b1s, b2s, b3s, b4s, b1m, b2m, b3m, b4m, b5, b6, b7, b8, b9 = [self.__coeffs.get(c) for c in ['b1s', 'b2s', 'b3s', 'b4s', 
               'b1m', 'b2m', 'b3m', 'b4m', 'b5', 'b6', 
               'b7', 'b8', 'b9']]
        
        pp = self.__data.columns[1:].to_numpy()
        T = self.__data.loc[:]['T'].to_numpy()
        vv = self.__data.drop(columns='T').to_numpy()
        
        Ttrans = self.__T_p_at_trans[:,0]
        
        self.__T_v_p_melt = np.empty((0,3))
        self.__T_v_p_solid = np.empty((0,3))
        
        for i, (p, ttrans, v_v) in enumerate(zip(pp, Ttrans, vv.T)):
            p = float(p)*1e5
            t = np.linspace(ttrans-20, ttrans+20, 500)
            f = self.func_v_melt_state(p, t, b1m, b2m, b3m, b4m,b5)
            g = self.func_v_solid_state(p,t,b5,b1s,b2s,b3s,b4s,b7,b8,b9)
            
            ttrans_new = t[(np.abs(f-g)).argmin()]
            self.__T_p_at_trans[i,0] =ttrans_new
            #print('At {} bar, ttrans = {} °C'.format(p/1e5, ttrans_new-273.15))
        
            
        self.fit_T_at_trans()
        b5_new, b6_new = [self.__coeffs.get(c) for c in coeffs]  
        
        res = np.abs((b5-b5_new)/b5) 
        self.split_data(pp, vv, T)                                                                         
        return res
     
    
    """
    Here the defined functions is combined in one method for fitting the pvT data
    """        
        
    def fit_pvT_curve(self, guess = 100,  **kwargs):
        
        options = {
                'modus' : 'differential_evolution',
                'Methods' : ['best1bin', 'randtobest1bin', ], 
                'maxInt' : 100, 
                'convergence' : 1e-05
                }
        options.update(kwargs)
        
        T = self.__data.loc[:]['T'].to_numpy()
        v = self.__data.drop(columns='T').to_numpy()
        p = self.__data.columns[1:].to_numpy()
        
        self.find_Ttrans(guess, dT = options['dT'])
        self.split_data(p, v, T)
        start_time = time.time()
        fit_ttrans = self.fit_T_at_trans()
        
        for i in range(options['maxInt']):
            fit_melt = self.fit_melt_state()
        
            if self.__polymer_type == 'sc':
                for m in options['Methods']:
                    try:
                        fit = self.fit_solid_state(m)
                        if fit.success:
                            break
                    except Exception as e:
                        print(f'------------------------\n with the strategy {m} we get an error: \n{e}\n')
                
            else:
                for m in options['Methods']:
                    try:
                        fit = self.fit_solid_state_amorph(m)
                        if fit.success:
                            break
                    except Exception as e:
                        print(f'------------------------\n with the strategy {m} we get an error: \n {e} \n')
            
            """
            Konvergence kriterium bestimmen und Fortschritt überprüfen
            """
            if i == 0:
                self.__coeffs_old.update(self.__coeffs)
            else:
                m =[]
                for key, value in self.__coeffs.items():
                    try:
                        m.append((value-self.__coeffs_old.get(key))/value)
                    except:
                        pass
                
                m_np = np.asarray(m)
                rel_error = np.sqrt(np.sum(m_np**2))
                
                
                self.__coeffs_old.update(self.__coeffs)
                #print(m_np)
                #print(rel_error)
            
            res = self.get_real_Ttrans()
            
            self.print_infos(i, time.time()-start_time, res)
            
            if res < options['convergence']:
                print('-------------------------------', end='') 
                print('calculation finished', end='')
                print('-------------------------------') 
                break
            
            self.split_data(p, v, T)
            for key, value in self.__coeffs.items():
                self.__dic_bounds_AMI[key] = (value*0.85, value*1.15)
            pass
    
    """
    This section is for ploting of the data and fitted curves to visualize the reuslt
    """
    def plot_data(self, ax):
        T = self.__data.loc[:]['T'].to_numpy()
        vv = self.__data.drop(columns='T').to_numpy()
        pp = self.__data.columns[1:].to_numpy()
        
        for p, v, color in zip(pp, vv.T, self.__color):
            ax.scatter(T, v, label = '{} bar'.format(p), s=1., color = color)
        
        ax.legend(loc='best')
        
    def plot_fit(self, ax):
        
        pp = self.__data.columns[1:].to_numpy()
        pp = np.append(pp, np.asarray([1]))
        

        
        for p, color in zip(pp, self.__color):
            p = float(p)*1e5

            T = np.linspace(300, 600, 500)
            
            ax.plot(T-273.15, self.func_pvT_master(p, T)*1e3, label='{} bar'.format(p/1e5), color= color, linewidth = 1)


    """
    Print the Fitting process infoprmation
    """
    def print_infos(self, step, c_time, error):
        if step == 0:
            print('Step \t runtime \t Relative error', end= '\t')
            #for key, values in self.__coeffs.items():
            #    print(' '+key+' ', end = '\t')
                
            print('')
            pass
        
        try:
            print('{:d} \t'.format(step)+time.strftime('%H:%M:%S', time.gmtime(c_time))+'\t {:.2e} '.format(error), end = '\t')
            #for key, values in self.__coeffs.items():
            #    print(' {:.2e} '.format(values), end = '\t')
            print('')
        except:
            #print('{:d} \t'.format(step)+time.strftime('%H:%M:%S', time.gmtime(c_time))+'\t {:.2e} \t {:.2e} \t {:.2e} \t {:.2e}'.format(e_ttrans, e_melt[0], e_solid, error))
            pass
    
    """
    some misc funtions to get some data within the class object, like coeffs, updating the bounds
    loding existing coeffizient data, ....
    """        
    def get_coeffs(self):
        return self.__coeffs
    
    def update_bounds(self, **kwargs):
        self.__dic_bounds_AMI.update(kwargs)
        pass
    
    def load_coeffs(self, fname):
        a = np.loadtxt(fname, dtype=np.str, delimiter=',')
        coeffs ={key:float(value) for key, value in a}
        self.__coeffs.update(coeffs)
        
if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    
    guess = 100
    dT = 50
    
    
    for fn in glob.glob(path+'\*.txt'):
        #fn = path +r'\PP_11.09.2019_1.txt'#r'/PP_11.09.2019_1.txt'
        #print(fn)
        fig, ax = plt.subplots(figsize=(4,3))
        
        pvT = pvT_analysis(fn, 'a')#'semi-crystalline') 
        
        pvT.plot_data(ax)
    
        if True:   
            pvT.fit_pvT_curve(guess=guess, dT = dT)
        
            coef = pvT.get_coeffs()
        
            c = np.asarray([ [key, value] for key, value in coef.items()])
            np.savetxt(fn.replace('.txt', 'coeffs.out'), c, delimiter = ',', fmt='%s')
        
        
        pvT.load_coeffs(fn.replace('.txt', 'coeffs.out'))

        pvT.plot_fit(ax)  
        
        ax.set(xlim=[20,290], ylim=[0.76,0.86])
        
        fig.savefig(fn.replace('.txt', '.png'), dpi = 600)
        break
       
        
    pass