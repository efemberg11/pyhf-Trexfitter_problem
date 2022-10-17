from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyhf
import json
import pyhf.readxml

def make_model(channel_list):
        spec["channels"] = [c for c in spec["channels"] if c["name"] in channel_list]

        w = pyhf.Workspace(spec)
        #print(w.modifiers)
        print('')
        #wstat = w._prune_and_rename(prune_modifier_types=['histosys', 'normsys'])#, 'staterror'])
        wstat = w
        print(wstat.modifiers)
        m = wstat.model(
            measurement_name="my_wrcspc", #COMPACT_13
            modifier_settings={
               "normsys": {"interpcode": "code4"},
               "histosys": {"interpcode": "code4p"},

            },
        )
        d = wstat.data(m)
        return wstat, m, d

def asimov_fitresults(chanel, constraints=None):
    _, model, data = make_model(chanel)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True, errordef=0.5, tolerance=0.01))

    constraints = constraints or []
    init_pars = model.config.suggested_init()
    fixed_params = model.config.suggested_fixed()
    bounds = model.config.suggested_bounds()
    bounds[model.config.poi_index] = [-5, 5]
    for idx, fixed_val in constraints:
        init_pars[idx] = fixed_val
        fixed_params[idx] = True

    mu_test = 1
    as_data = pyhf.infer.calculators.generate_asimov_data(
        mu_test, 
        data, 
        model, 
        init_pars=init_pars,
        fixed_params=fixed_params,
        par_bounds=bounds)
    
    result = pyhf.infer.mle.fit(
        as_data,
        model,
        init_pars=init_pars,
        fixed_params=fixed_params,
        par_bounds=bounds,
        return_uncertainties=True,
    )
    bestfit = result[:, 0]
    errors = result[:, 1]
    return model, data, bestfit, errors

def fitresults(chanel, constraints=None):
    _, model, data = make_model(chanel)

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True, errordef=0.5, tolerance=0.01))

    constraints = constraints or []
    init_pars = model.config.suggested_init()
    fixed_params = model.config.suggested_fixed()
    bounds = model.config.suggested_bounds()
    bounds[model.config.poi_index] = [-5, 5]
    for idx, fixed_val in constraints:
        init_pars[idx] = fixed_val
        fixed_params[idx] = True

    result = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        fixed_params=fixed_params,
        par_bounds=bounds,
        return_uncertainties=True,
    )
    bestfit = result[:, 0]
    errors = result[:, 1]
    return model, data, bestfit, errors

chanal = ["three_lep_presel_1jet", "three_lep_presel_2jets", "three_lep_presel_atLeast_3jets"]

#spec = pyhf.readxml.parse('COMPACT_13/RooStats/COMPACT_13.xml', Path.cwd())
spec = pyhf.readxml.parse('my_wrcspc/RooStats/my_wrcspc.xml', Path.cwd())
model,data,bestfit,errors = asimov_fitresults('three_lep_presel_1jet')
print(model.config.channels, ' ', model.config.poi_name, ' ', bestfit[model.config.poi_index], " ", errors[model.config.poi_index])
