# pyhf-Trexfitter_problem
Hi!
I'm trying to compare fit results in pyhf and trexfitter using same workspace.
Bestfit values have good compability, but errors have differences for both fit cases (asimov fit and data fit).
Such differences was observed for pyhf (v. 0.5.4 and 0.7.0) vs trex fitter v4.15.

To get pyhf results, run: pyhf_fit.py.
To get trex fitter results, run: trex-fitter f clear.config "Regions=three_lep_presel_1jet"

In trexfitter's clear.config file, you can change FitBlind value to TRUE for asimov fit, or FALSE for data fit.

Maybe there are some parameters, which mismatching in pyhf or in trexfitter. 
