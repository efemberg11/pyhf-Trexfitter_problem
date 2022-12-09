
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyhf
import pyhf.readxml
import time
import iminuit

begin_o = time.time()

chanal = [  'three_lep_presel_1jet'
            #'SR_WVZ_NJ1',
            #'SR_WVZ_NJ2',
            #'SR_WVZ_NJ3'
            ]#, 'ttZ_3L_CR']
meas_name = 'my_wrcspc'
            #'WVZ1_3LJs_Clean_NoSyst_NJsInclTrain_VS05_NT400_SplusB_AddSyst_WZ1W'

spec = pyhf.readxml.parse(meas_name+'/RooStats/'+meas_name+'.xml', Path.cwd(),)# track_progress=True)


#print(spec)

def make_model(channel_list):
    spec["channels"] = [c for c in spec["channels"] if c["name"] in channel_list]

    w = pyhf.Workspace(spec)
    m = w.model(
        measurement_name=meas_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    d = w.data(m)
    return w, m, d

def fitresults(constraints=None):
    _, model, data = make_model(chanal)

    tolerance = 0.01
    strategy = 0
    errordef = 1


    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True,
                                                            #tolerance=tolerance,
                                                            errordef=errordef,
                                                            #strategy=strategy,
                                                            maxiter = 30000))#, tolerance = 0.0001))

    constraints = constraints or []
    init_pars = model.config.suggested_init()
    fixed_params = model.config.suggested_fixed()
    bounds = model.config.suggested_bounds()
    #bounds[41] = [0, 2]
    #init_pars[41]=0
    #print(model.config.poi_index)
    for idx, fixed_val in constraints:
        init_pars[idx] = fixed_val
        fixed_params[idx] = True

    result, result_obj = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        fixed_params=fixed_params,
        par_bounds=bounds,
        return_uncertainties=True,
        return_result_obj=True,        
    )
    bestfit = result[:, 0]
    errors = result[:, 1]

    #labels = model.config.par_names
    #a=result_obj.minuit.minos('x'+str(model.config.poi_index))
    #print(a)

    return model, data, bestfit, errors

    
def calc_impact(idx, b, e, i, width, poi_index):
    _, _, bb, ee = fitresults([(idx, b + e)])
    poi_up_post = bb[poi_index]

    _, _, bb, ee = fitresults([(idx, b - e)])
    poi_dn_post = bb[poi_index]

    _, _, bb, ee = fitresults([(idx, b + width)])
    poi_up_pre = bb[poi_index]

    _, _, bb, ee = fitresults([(idx, b - width)])
    poi_dn_pre = bb[poi_index]
    return np.asarray([poi_dn_post, poi_up_post, poi_dn_pre, poi_up_pre])

def get_impact_data():
    model, _, b, e = fitresults()
    
    widths = pyhf.tensorlib.concatenate(
        [
            model.config.param_set(k).width()
            if model.config.param_set(k).constrained
            else [0] * model.config.param_set(k).n_parameters
            for k, v in model.config.par_map.items()
        ]
    )
    initv = pyhf.tensorlib.concatenate(
        [
            model.config.param_set(k).suggested_init
            for k, v in model.config.par_map.items()
        ]
    )
    labels = np.asarray(
        [
            f"{k}[{i:02}]" if model.config.param_set(k).n_parameters > 1 else k
            for k in model.config.par_order
            for i in range(model.config.param_set(k).n_parameters)
        ]
    )
    poi_free = b[model.config.poi_index]
    impacts = []
    
    for i, width in enumerate(widths):
        if width is None:
            impacts.append(0)
        if i == model.config.poi_index:
            continue
        if i ==41:
            impacts.append([0, 0, 0, 0])
            continue
        if i ==21:
            impacts.append([0, 0, 0, 0])
            continue
        if i % 5 == 0:
            print(i)
        impct = calc_impact(i, b[i], e[i], initv[i], width, model.config.poi_index)
        impacts.append(impct - poi_free)
    return np.asarray(impacts), labels



model, data, bestfit, errors = fitresults()


pulls = pyhf.tensorlib.concatenate(
    [
        (bestfit[model.config.par_slice(k)] - model.config.param_set(k).suggested_init)
        / model.config.param_set(k).width()
        for k in model.config.par_order
        if model.config.param_set(k).constrained
    ]
)
pullerr = pyhf.tensorlib.concatenate(
    [
        errors[model.config.par_slice(k)] / model.config.param_set(k).width()
        for k in model.config.par_order
        if model.config.param_set(k).constrained
    ]
)



impacts,labels = get_impact_data()
# # print(impacts)
# # print(b)

impcord  = np.argsort(np.max(np.abs(impacts[:,:2]),axis=1))
simpacts = impacts[impcord]
bestfit = bestfit[impcord]
slabels   = labels[impcord]
serrors = errors[impcord]
#pulls = pulls[impcord]
#pullerr = pullerr[impcord]

for idx in range(len(slabels)):
        print(f' {idx} {slabels[idx]} bestfit {bestfit[idx]:.7f} post_dn {simpacts[idx,0]:.5f} post_up {simpacts[idx,1]:.5f} pre_dn {simpacts[idx,2]:.5f} pre_post {simpacts[idx,3]:.5f} ') 


# df = pd.DataFrame()
# df = pd.DataFrame(pulls, columns=['Pulls'], index = slabels)
# df.insert(1, 'Pull_err', pullerr)
# df.insert(2,'poi_dn_post', np.asarray(simpacts)[:,0])
# df.insert(3,'poi_up_post', np.asarray(simpacts)[:,1])
# df.insert(4,'poi_dn_pre', np.asarray(simpacts)[:,2])
# df.insert(5,'poi_up_pre', np.asarray(simpacts)[:,3])
# df.insert(6, 'bestfit', bestfit)
stop=time.time()
# df.to_csv('pyhf_out_all_ch.csv')
# print(df.to_string())
print(f'time is {stop-begin_o }')