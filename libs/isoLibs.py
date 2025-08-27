#%% Importing
# numericals
import numpy as np
import json, zlib, base64

# scipy
from scipy.integrate import trapz
from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import differential_evolution
# files handling
import os
import pickle

# Graph
import matplotlib.pyplot as plt

#%% Single Isotherm 1: isotherm models
def Arrh(T, dH, T_ref):
    exp_term = np.exp(np.abs(dH)/8.3145*(1/T-1/T_ref))
    return exp_term

def iso2isoArr(iso_P_only, dH,T_ref):
    iso_PT = lambda P,T: iso_P_only(Arrh(T,dH,T_ref)*P)
    return iso_PT

def iso2isoT(iso_P_only):
    iso_PT = lambda P,T: iso_P_only(P)
    return iso_PT

### With 2 parameters ###
def Lang(par,P): # Langmuir isotherm model
    bP = par[1]*P
    deno = 1+ bP
    nume = par[0]*bP
    q = nume/deno
    return q

def Freu(par, P): # Freundlich isotherm model
    q = par[0]*P**par[1]
    return q

### With 3 parameters ###
def Quad(par,P): # Quadratic isotherm model
    bP = par[1]*P
    dPP = par[2]*P**2
    deno = 1+ bP + dPP
    nume = par[0]*(bP + 2*dPP)
    q = nume/deno
    return q

def Sips(par,P): # Sips isotherm model 
    P = np.maximum(P, 1e-8)
    n = par[2]
    numo = par[0]*par[1]*P**n
    deno = 1 + par[1]*P**n
    q = numo/deno
    return q

### With 4 parameters ###
def DSLa(par,P):
    nume1 = par[0]*par[1]*P
    deno1 = 1+par[1]*P
    nume2 = par[2]*par[3]*P
    deno2 = 1+par[3]*P
    q = nume1/deno1 + nume2/deno2
    return q

iso_fn_candidates = [
        [Lang, Freu,],
        [Quad, Sips,],
        [DSLa,]
        ]

iso_fn_candidate_index = [
        ['Lang', 'Freu'],
        ['Quad', 'Sips'],
        ['DSLa']
        ]
iso_par_num = [2,3,4]

#%% Single Isothemr 2: Objective function
def iso2err(par,P,q,iso_fn):
    par_arr = np.array(par)
    is_nega = par_arr<0
    penaltyy = np.sum(par_arr[is_nega]**2)*50
    par_arr[is_nega] = 0
    
    #diff = (iso_fn(par_arr,np.array(P)) - np.array(q))/(np.array(q)+1E-3)
    diff = iso_fn(par_arr,np.array(P)) - np.array(q)
    err_sum = np.mean(diff**2)
    return err_sum + penaltyy

#%% Single Isothemr 3: Fitting function with a single model
method_list = ['Nelder-mead','Powell','COBYLA','shgo','differential_evolution']
def find_par(isofn, n_par, P,q, methods):
    p_arr = np.array(P)
    q_arr = np.array(q)
    obj_fn = lambda par: iso2err(par,p_arr,q_arr,isofn)
    optres_fun = []
    optres_x = []
    for me in methods:
        optres_tmp = []
        if me =='shgo':
            bounds = np.zeros([n_par,2])
            bounds[:,1] = 5
            optres_tmp = shgo(obj_fn,bounds,)
        elif me == 'differential_evolution':
            bounds = np.zeros([n_par, 2])
            bounds[:,1] = 5
            optres_tmp = differential_evolution(obj_fn, bounds,)
        else:
            x0 = 2*np.ones(n_par)  # INITIAL GUESS !!!
            x0[0] = q[-1]
            optres_tmp = minimize(obj_fn,x0,method = me)
        optres_fun.append(optres_tmp.fun)
        optres_x.append(optres_tmp.x)
    bestm = np.argmin(optres_fun)
    par_sol = optres_x[bestm]
    fn_sol = optres_fun[bestm]
    return par_sol, fn_sol, optres_x, optres_fun

#%% Single Isotherm 4: Fitting with diff. isotherm models
def best_isomodel(P, q, iso_par_nums = [2, 3, 4], 
iso_fun_lists = None, iso_fun_index = None, tol = 1.0E-5):
    if iso_fun_lists == None:
        iso_fun_lists = []
        for ii in iso_par_nums:
            if ii == 2:
                iso_fun_lists.append([Lang, Freu,])
            if ii == 3:
                iso_fun_lists.append([Quad, Sips,])
            if ii == 4:
                iso_fun_lists.append([DSLa,])
    if iso_fun_index == None:
        iso_fun_index = []
        iso_fun_index.append(["Langmuir","Freundlich"])
        iso_fun_index.append(["Quadratic","Sips"])
        iso_fun_index.append(["Dual-site Langmuire"])
        '''
        for isolii in iso_fun_lists:
            indx_tmp = list(range(len(isolii)))
            iso_fun_index.append(indx_tmp)
        '''
    optfn_list = []
    optx_list = []
    iso_best_list = []
    iso_best_indx = []
    for n_par,iso_funs,iso_ind in zip(iso_par_nums, iso_fun_lists, iso_fun_index):
        ########################################
        parsol_list_tmp = []
        fnsol_list_tmp = []
        for isof in iso_funs:
            par_sol_tmp, fn_sol_tmp, _, _ = find_par(isof, n_par, P, q, method_list)
            parsol_list_tmp.append(par_sol_tmp)
            fnsol_list_tmp.append(fn_sol_tmp)
        if len(fnsol_list_tmp) < 2:
            arg_best = 0
        else:
            arg_best = np.argmin(np.array(fnsol_list_tmp))
        
        parsol_tmp = np.array(parsol_list_tmp)[arg_best]
        fnsol_tmp = np.array(fnsol_list_tmp)[arg_best]
        isobest_tmp = np.array(iso_funs)[arg_best]
        isobest_indx = np.array(iso_ind)[arg_best]

        optx_list.append(parsol_tmp)
        optfn_list.append(fnsol_tmp)
        iso_best_list.append(isobest_tmp)
        iso_best_indx.append(isobest_indx)
        if fnsol_tmp/len(q) <= tol:
            break
    #print(optfn_list)
    argMIN = np.argmin(np.array(optfn_list))
    x_best = np.array(optx_list, dtype=np.ndarray)[argMIN]
    iso_best = lambda pp: iso_best_list[argMIN](x_best, pp)
    str_best = iso_best_indx[argMIN]
    fnval_best = np.array(optfn_list)[argMIN]
    return iso_best, x_best, str_best, fnval_best
    #return iso_best, x_best , fnval_best

#%% Fitting for different T (Heat of adsorption)

def fit_diffT(p_list, q_list, T_list, i_ref,
        iso_par_nums = [2, 3, 4], 
        iso_fun_lists = None, 
        iso_fun_index = None,
        tol = 1.0E-5):

    p_ref = p_list[i_ref]
    q_ref = q_list[i_ref]
    fit_res_ref = best_isomodel(p_ref,q_ref, 
            iso_par_nums, iso_fun_lists, iso_fun_index, tol)
    iso_ref, param_ref, model_ref, fnval_ref = fit_res_ref
    #print(model_ref)
    #print(fit_res_ref)
    n_da = len(T_list)
    theta_list = []
    p_norm = []
    q_norm = []
    # Objective function for theta
    for kkk in range(n_da):
        if kkk == i_ref:
            theta_list.append(1)
            p_norm.append(p_list[kkk])
            q_norm.append(q_list[kkk])
            continue
        def err_theta(theta):
            pp_tmp = p_list[kkk]*theta
            qq_tmp = q_list[kkk]
            qq_pred = iso_ref(pp_tmp)
            diff = (qq_tmp - qq_pred)
            mse = np.mean(diff**2)
            return mse
        # Find theta
        opt_tmp_theta = minimize(
            err_theta, 0.9,
            method = 'Nelder-Mead')
        theta_list.append(opt_tmp_theta.x[0])
        p_norm.append(p_list[kkk]*opt_tmp_theta.x[0])
        q_norm.append(q_list[kkk])
    # Find heat of adsorption
    T_ref = T_list[i_ref]
    T_ref_arr = T_ref*np.ones_like(T_list)
    T_data_arr = np.array(T_list)
    def err_dH(dH):
        theta_pred = np.exp(dH/8.31446*(1/T_data_arr - 1/T_ref_arr))
        err_theta = theta_pred - np.array(theta_list)
        err_sqr_mean = np.mean(err_theta**2)
        return err_sqr_mean
    opt_res_dH = minimize(
        err_dH,20000, method = 'Nelder-Mead')
    if opt_res_dH.fun > 0.1:
        opt_res_dH_pre = opt_res_dH
        opt_res_dH = minimize(
            err_dH,20000, method = 'Powell')
        if opt_res_dH_pre.fun < opt_res_dH.fun:
            opt_res_dH = opt_res_dH_pre
    #print(opt_res_dH)
    dH = opt_res_dH.x[0]

    p_norm_arr = np.concatenate(p_norm)
    q_norm_arr = np.concatenate(q_norm)
    fit_res_all = best_isomodel(p_ref,q_ref, 
            iso_par_nums, iso_fun_lists, iso_fun_index, tol)

            #p_norm_arr,q_norm_arr, tol = 1E-5)
    iso_ref = fit_res_all[0]
    err_fit_all = fit_res_all[3]
    iso_params = fit_res_all[1]
    str_best = fit_res_all[2]
    R_gas = 8.3145
    iso_all = lambda P_in, T_in : np.reshape(iso_ref(P_in*np.exp(dH/R_gas*(1/T_in - 1/T_ref))), [-1,])
    iso_all = lambda P_in, T_in : iso_ref(P_in*np.exp(dH/R_gas*(1/T_in - 1/T_ref)))
    #print(err_fit_all)
    #print(fit_res_all[2])
    #q_pre_norm = iso_ref(p_norm_arr)
    #diff_all = (q_pre_norm - q_norm_arr)/(q_norm_arr+1E-3)
    #err_fit_all = np.mean(diff_all**2)
    var_return = (iso_all, iso_params, str_best, err_fit_all,
            dH, T_ref, theta_list)
    return var_return
    #return iso_all, fnval_ref, theta_list, err_fit_all, dH, T_ref, iso_params

#%% RAST 1: (ln \gamma)
def ln_gamma_i(x,Lamb, C, piA_RT):
    N = len(x)
    ln_gamma = np.zeros(N)   # array N
    sum_xlam = []   # array N
    sum_xlam_ov_xlam = []
    exp_term = []
    for ii in range(N):
        xlam_tmp = 0
        for jj in range(N):
            xlam_tmp = xlam_tmp + x[jj]*Lamb[ii,jj]
        sum_xlam.append(xlam_tmp)
        sum_xlam_xlam = 0
        for kk in range(N):
            sum_xlam_tmp = 0
            for ll in range(N):
                sum_xlam_tmp = sum_xlam_tmp + Lamb[kk,ll]*x[ll]
            sum_xlam_xlam = sum_xlam_xlam + Lamb[kk,ii]*x[kk]/sum_xlam_tmp
        sum_xlam_ov_xlam.append(sum_xlam_xlam)
        exp_term_tmp = 1-np.exp(-C*piA_RT)
        exp_term.append(exp_term_tmp)
        all_term_tmp = (1-np.log(xlam_tmp)-sum_xlam_xlam)*exp_term_tmp
        ln_gamma[ii] = all_term_tmp
    return ln_gamma

#%% RAST 2: RAST mixture isotherm model
def rast(isotherm_list,P_i,T, Lamb, C):
    if len(Lamb.shape) != 2:
        print('Lambda should be N x N array or matrix!')
        return
    elif Lamb.shape[0] != Lamb.shape[1]:
        print('Lambda should be N x N array or matrix!')
        return
    else:
        N = Lamb.shape[0]
    '''
    def spreading_pressure(iso, P_max):
        P_ran = np.linspace(0.0001,P_max)
        q_ov_P = iso(P_ran)/P_ran
        spr_P = trapz(q_ov_P, P_ran)
        return spr_P
    '''
    iso_list = []
    #iso_spr = []
    for isoo in isotherm_list:
        iso_tmp = lambda pp: isoo(pp, T)
        iso_list.append(iso_tmp)
        #iso_spr_tmp = lambda ppp: spreading_pressure(iso_tmp, ppp)
        #iso_spr.append(iso_spr_tmp)
    def spr_press(isoPT, P_max,T):
        P_ran = np.linspace(0.0001,P_max)
        q_ov_P = isoPT(P_ran,T)/P_ran
        spr_P = trapz(q_ov_P, P_ran)
        return spr_P

    def spreading_P_err(x_N_piART):
        xx_first = x_N_piART[:N-1]
        xx_last = [1-np.sum(xx_first)]
        xx = np.concatenate([xx_first, xx_last])
        rms_err = 0
        for ii in range(N):
            if xx[ii] <0.0001:
                rms_err = rms_err+50*xx[ii]**2
                xx[ii] = 0.0001
            elif xx[ii] > 0.999:
                rms_err = rms_err+50*(xx[ii]-1)**2
                xx[ii] = 0.9999
        
        spr_P = x_N_piART[-1]
        ln_gam = ln_gamma_i(xx,Lamb, C, spr_P,)
        gamm = np.exp(ln_gam)
        Po_i = P_i/gamm/xx
        spr_P_new = np.zeros(N)
        for ii in range(N):
            #spr_P_tmp = iso_spr[ii](Po_i[ii])
            spr_P_tmp = spr_press(isotherm_list[ii],Po_i[ii],T)
            spr_P_new[ii] = spr_P_tmp
        rms_err = rms_err + np.sum((spr_P_new - spr_P)**2)
        return rms_err
    
    y_i = P_i/np.sum(P_i)
    x_init = P_i/np.sum(P_i)
    x_init = x_init[:-1]
    
    piA_RT_list = []
    qm = []
    #theta = []
    bP = []
    for iso,pp in zip(isotherm_list,P_i):
        P_ran = np.linspace(0.0001,pp)
        q_P = iso(P_ran, T)/P_ran
        piA_RT_tmp = trapz(q_P,P_ran)
        piA_RT_list.append(piA_RT_tmp)
        q_max = iso(1E8, T)
        theta_tmp = q_P[-1]*pp/q_max
        bP_tmp = theta_tmp/(1-theta_tmp)
        bP.append(bP_tmp)
        #theta.append(theta_tmp)
        qm.append(q_max)
    bP_sum = np.sum(bP)
    q_extended = np.array(qm)*np.array(bP)/(1+ bP_sum)
    x_extended = q_extended/np.sum(q_extended)
    x_ext_init = x_extended[:-1]

    opt_list = []
    opt_x_list = []
    opt_fn_list = []
    for spr_P0 in piA_RT_list:
        x0 = np.concatenate([x_init, [spr_P0]])
        optres_tmp = minimize(spreading_P_err, x0, method = 'Nelder-mead')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-6:
            #print(optres_tmp.fun)
            #print('YEAH')
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'Powell')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'COBYLA')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
    for spr_P0 in piA_RT_list:
        x0 = np.concatenate([x_ext_init, [spr_P0]])
        optres_tmp = minimize(spreading_P_err, x0, method = 'Nelder-mead')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'Powell')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'COBYLA')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
    #print(opt_fn_list)
    arg_min = np.argmin(opt_fn_list)
    x_re = np.zeros(N)
    x_re[:-1] = opt_list[arg_min].x[:-1]
    x_re[-1] = np.min([1- np.sum(x_re[:-1],0)])
    piA_RT_re = opt_list[arg_min].x[-1]
    ln_gam_re = ln_gamma_i(x_re,Lamb, C, piA_RT_re)
    gamma_re = np.exp(ln_gam_re)
    #print(iso_spr[0](P_i[0]/optres_tmp.x[0]/gamma_re[0]))
    #print(iso_spr[1](P_i[1]/(1- optres_tmp.x[0])/gamma_re[1]))
    arg_0 = x_re == 0
    arg_non0 = arg_0 == False
    P_pure = np.zeros(N)
    P_pure[arg_non0] = np.array(P_i)[arg_non0]/x_re[arg_non0]/gamma_re[arg_non0]
    #P_pure[arg_0] = np.array(P_i)[arg_0]/x_re[arg_0]/gamma_re[arg_0]
    #P_pure[arg_0] = 0
    q_pure = np.zeros(N)
    for ii in range(N):
        q_pure[ii] = iso_list[ii](P_pure[ii])
    q_tot = 1/(np.sum(x_re/q_pure))
    q_return = q_tot*x_re
    return q_return

# %% IAST funciton
def IAST(isotherm_list, P_i, T):
    if len(isotherm_list) != len(P_i):
        print('# of funcitons in "isotherm_list" should match the dimension of "P_i!"')
        return
    else:
        N = len(isotherm_list)

    iso_list = []
    for isoo in isotherm_list:
        iso_tmp = lambda pp: isoo(pp, T)
        iso_list.append(iso_tmp)
    def spr_press(isoPT, P_max, T):
        P_ran = np.linspace(0.0001, P_max, 101)
        q_ov_P = isoPT(P_ran, T)/P_ran
        spr_P = trapz(q_ov_P, P_ran)
        return spr_P
    def spreading_P_err(x_N_piART):
        xx_first = x_N_piART[:N-1]
        xx_last  = [1-np.sum(xx_first)]
        xx = np.concatenate([xx_first, xx_last])
        rms_err = 0
        for ii in range(N):
            if xx[ii] < 0.0001:
                rms_err = rms_err + 50*xx[ii]**2
                xx[ii] = 0.0001
            elif xx[ii] > 0.9995:
                rms_err = rms_err + 50*(0.9995 - xx[ii])**2
                xx[ii] = 0.9995
        spr_P = x_N_piART[-1]
        Po_i = P_i/xx
        spr_P_new = np.zeros(N)
        for ii in range(N):
            spr_P_tmp = spr_press(isotherm_list[ii], Po_i[ii], T)
            spr_P_new[ii] = spr_P_tmp
        rms_err = rms_err + np.sum((spr_P_new - spr_P)**2)
        return rms_err

    y_i = P_i/np.sum(P_i)
    x_init = P_i/np.sum(P_i)
    x_init = x_init[:-1]

    # Initial guess: x_ext_init, piA_RT_list
    piA_RT_list = []
    qm = []
    bP = []
    for iso, pp in zip(isotherm_list, P_i):
        P_ran = np.linspace(0.0001, pp, 101)
        q_P = iso(P_ran, T)/P_ran
        piA_RT_tmp = trapz(q_P, P_ran)
        piA_RT_list.append(piA_RT_tmp)
        q_max = iso(1E8, T)
        theta_tmp = q_P[-1]*pp/q_max
        bP_tmp = theta_tmp*(1-theta_tmp)
        bP.append(bP_tmp)
        qm.append(q_max)
    bP_sum = np.sum(bP)
    q_extended = np.array(qm)*np.array(bP)/(1+bP_sum)
    x_extended = q_extended/np.sum(q_extended)
    x_ext_init = x_extended[:-1]

    opt_list = []
    opt_x_list = []
    opt_fn_list = []
    for spr_P0 in piA_RT_list:
        x0 = np.concatenate([x_init, [spr_P0]])
        # Nelder-Mead
        optres_tmp = minimize(spreading_P_err,
        x0, method = 'Nelder-mead')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-6:
            break
        # Powell
        optres_tmp = minimize(spreading_P_err, x0,
        method = 'Powell')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        # COBYLA
        optres_tmp = minimize(spreading_P_err,
        x0, method =  'COBYLA')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        # Differential evolution
        bound_de = [[0,1],]*len(x0)
        bound_de[-1][-1] = 200
        optres_tmp = differential_evolution(spreading_P_err,
        bounds = bound_de)
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
    arg_min = np.argmin(opt_fn_list)
    x_re = np.zeros(N)
    x_re[:-1] = opt_list[arg_min].x[:-1]
    x_re[-1] = np.min([1- np.sum(x_re[:-1], 0)])
    piA_RT_re = opt_list[arg_min].x[-1]
    arg_0 = x_re == 0
    arg_non0 = arg_0 == False
    P_pure = np.zeros(N)
    P_pure[arg_non0] = np.array(P_i)[arg_non0]/x_re[arg_non0]
    q_pure = np.zeros(N)
    for ii in range(N):
        q_pure[ii] = iso_list[ii](P_pure[ii])
    q_tot = 1/(np.sum(x_re/q_pure))
    q_return = q_tot*x_re
    return q_return            


# ---------- SERIALIZAR ----------
def _pack_isoFuncs(isoFuncs: dict) -> str:
    """
    Convierte IsoFuncs en una cadena base64 (JSON comprimido con zlib).
    No serializa lambdas; guarda nombre de modelo, parámetros, dH, Tref, etc.
    """
    payload = {
        "schema": "IsoFuncs.v1",
        "species": isoFuncs["species"],
        "iso_fun_name": isoFuncs["iso_fun_name"],
        "iso_fun_par": [list(map(float, p)) for p in isoFuncs["iso_fun_par"]],
        "iso_qs0": list(map(float, np.asarray(isoFuncs["iso_qs0"]).ravel())),
        "iso_dH": [None if v is None else float(v) for v in isoFuncs["iso_dH"]],
        "iso_Tref": [None if v is None else float(v) for v in isoFuncs["iso_Tref"]],
        "err_mse": list(map(float, isoFuncs.get("err_mse", [np.nan]*len(isoFuncs["species"])))),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    b64  = base64.b64encode(comp).decode("ascii")
    return b64

# ---------- DESERIALIZAR ----------
def _unpack_isoFuncs(encoded_str: str) -> dict:
    """
    Decodifica la cadena base64 y reconstruye IsoFuncs, incluyendo las funciones q(P,T).
    """
    model_map = {
        "Langmuir": Lang,
        "Freundlich": Freu,
        "Quadratic": Quad,
        "Sips": Sips,
        "Dual-site Langmuir": DSLa,
        "Dual-site Langmuire": DSLa,  # tolerancia a nombre
    }

    comp = base64.b64decode(encoded_str.encode("ascii"))
    raw  = zlib.decompress(comp)
    data = json.loads(raw.decode("utf-8"))

    if data.get("schema") != "IsoFuncs.v1":
        raise ValueError(f"Esquema no soportado: {data.get('schema')}")

    species      = data["species"]
    fun_names    = data["iso_fun_name"]
    fun_pars     = data["iso_fun_par"]
    qs0          = np.array(data["iso_qs0"], dtype=float)
    dH_list      = data["iso_dH"]
    Tref_list    = data["iso_Tref"]
    err_mse_list = data.get("err_mse", [np.nan]*len(species))

    fun_list = []
    for name, pars, dH, Tref in zip(fun_names, fun_pars, dH_list, Tref_list):
        if name not in model_map:
            raise ValueError(f"Modelo no implementado: {name}")
        base_fn = model_map[name]
        pars_np = np.array(pars, dtype=float)

        # Si hay dependencia con T (dH y Tref), usa iso2isoArr; si no, solo P.
        if dH is None or Tref is None:
            iso_fun_PT = (lambda P, T, base=base_fn, pp=pars_np: base(pp, P))
        else:
            iso_fun_PT = iso2isoArr(lambda P, base=base_fn, pp=pars_np: base(pp, P),
                                    dH=float(dH), T_ref=float(Tref))

        fun_list.append(iso_fun_PT)

    isoFuncs = {
        "species": species,
        "iso_fun": fun_list,             # funciones q(P,T) reconstruidas
        "iso_fun_name": fun_names,
        "iso_fun_par": [np.array(p) for p in fun_pars],
        "iso_qs0": qs0,
        "iso_dH": dH_list,
        "iso_Tref": Tref_list,
        "err_mse": err_mse_list,
    }
    return isoFuncs    