"""Known-parameter values for the motivating example (simulation constructions A and B,
matched D2 = .80 at the common schedule t = 0,1,2,3 centered at 1.5). No fitting."""
import json
import numpy as np

T = np.array([0., 1, 2, 3]) - 1.5
SETTINGS = {  # from contamination_pilot.py build_settings(g00=100, target_d2=.80)
    "A": dict(sigma2=1.25, G00=100.0, G0s=0.0, tau2=1.0),
    "B": dict(sigma2=20.0, G00=100.0, G0s=float(np.sqrt(393.75 / 4.75)), tau2=1.0),
}

def quantities(sigma2, G00, G0s, tau2, t=T, c=0.5):
    n = len(t); Z = np.column_stack([np.ones(n), t]); G = np.array([[G00, G0s], [G0s, tau2]])
    V = Z @ G @ Z.T + sigma2 * np.eye(n)
    w = np.linalg.solve(V, Z @ G[:, 1]); D2 = w @ Z @ G[:, 1] / tau2
    SS = ((t - t.mean()) ** 2).sum(); J = SS / sigma2; R = tau2 * J / (1 + tau2 * J)
    S = G00 - G0s ** 2 / tau2
    kappa = (w.sum() ** 2) * S / (tau2 * D2)
    resid_share = sigma2 * (w @ w) / (tau2 * D2)
    k = G0s / tau2 + t.mean(); E = S + sigma2 / n
    prec = dict(prior=1 / tau2, within_slope=SS / sigma2, person_mean=k ** 2 / E)
    return dict(D2=D2, R_direct=R, Delta_fit=D2 - R, kappa=kappa, sqrt_kappa=np.sqrt(kappa),
                resid_share=resid_share, target_share=D2, rho=G0s / np.sqrt(G00 * tau2), S=S,
                k=k, precision=prec, corr_at_c=c * np.sign(w.sum()) * np.sqrt(kappa),
                check_composition=D2 + kappa + resid_share)

if __name__ == "__main__":
    out = {name: quantities(**p) for name, p in SETTINGS.items()}
    print(json.dumps(out, indent=2, default=float))
    json.dump(out, open("fig1_values.json", "w"), indent=2, default=float)
