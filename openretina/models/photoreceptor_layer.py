"""
Photoreceptor layer code adapted from https://github.com/saadidrees/dynret/blob/main/models.py

See https://www.biorxiv.org/content/10.1101/2023.06.20.545728v1 for the original paper.
Also: https://www.biorxiv.org/content/10.1101/2021.02.13.431101v1.full for the original biophysical model.
"""

from typing import Optional

import torch
import torch.nn as nn

PR_PARAMS = {
    # Opsin decay rate constant
    "sigma": 2.2,
    "sigma_scaleFac": 10.0,
    "sigma_trainable": False,
    # PDE decay rate constant (in the original paper is constrained equal to sigma)
    "phi": 2.2,
    "phi_scaleFac": 10.0,
    "phi_trainable": False,
    # PDE dark activation rate
    "eta": 2.0,
    "eta_scaleFac": 1000.0,
    "eta_trainable": True,
    # Ca2+ extrusion rate constant
    "beta": 0.9,
    "beta_scaleFac": 10.0,
    "beta_trainable": False,
    # cGMP-to-current constant
    "cgmp2cur": 0.01,
    "cgmp2cur_scaleFac": 1.0,
    "cgmp2cur_trainable": False,
    # cGMP channel cooperativity
    "cgmphill": 3.0,
    "cgmphill_scaleFac": 1.0,
    "cgmphill_trainable": True,
    #
    "cdark": 1.0,
    "cdark_scaleFac": 1.0,
    "cdark_trainable": True,
    # Channel-feedback decay rate constant
    "betaSlow": 0.4,
    "betaSlow_scaleFac": 1.0,
    "betaSlow_trainable": True,
    # Ca2+ GC-cooperativity
    "hillcoef": 4.0,
    "hillcoef_scaleFac": 1.0,
    "hillcoef_trainable": True,
    # Ca2+ GC-affinity
    "hillaffinity": 0.5,
    "hillaffinity_scaleFac": 1.0,
    "hillaffinity_trainable": True,
    # Opsin gain
    "gamma": 1.0,
    "gamma_scaleFac": 10.0,
    "gamma_trainable": True,
    #
    "gdark": 0.35,
    "gdark_scaleFac": 100.0,
    "gdark_trainable": True,
    #
    "timeBin": 5,
}


class PhotoreceptorLayer(nn.Module):
    def __init__(self, pr_params: Optional[dict], units=1):
        super().__init__()
        self.units = units
        pr_params = pr_params if pr_params is not None else PR_PARAMS
        self.pr_params = pr_params

        dtype = pr_params["dtype"] if "dtype" in pr_params else torch.float32

        self.sigma = nn.Parameter(
            torch.tensor(pr_params["sigma"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["sigma_trainable"],
        )
        self.sigma_scaleFac = nn.Parameter(
            torch.tensor(pr_params["sigma_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.phi = nn.Parameter(
            torch.tensor(pr_params["phi"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["phi_trainable"],
        )
        self.phi_scaleFac = nn.Parameter(
            torch.tensor(pr_params["phi_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.eta = nn.Parameter(
            torch.tensor(pr_params["eta"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["eta_trainable"],
        )
        self.eta_scaleFac = nn.Parameter(
            torch.tensor(pr_params["eta_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.beta = nn.Parameter(
            torch.tensor(pr_params["beta"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["beta_trainable"],
        )
        self.beta_scaleFac = nn.Parameter(
            torch.tensor(pr_params["beta_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.cgmp2cur = nn.Parameter(
            torch.tensor(pr_params["cgmp2cur"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["cgmp2cur_trainable"],
        )

        self.cgmphill = nn.Parameter(
            torch.tensor(pr_params["cgmphill"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["cgmphill_trainable"],
        )
        self.cgmphill_scaleFac = nn.Parameter(
            torch.tensor(pr_params["cgmphill_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.cdark = nn.Parameter(
            torch.tensor(pr_params["cdark"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["cdark_trainable"],
        )

        self.betaSlow = nn.Parameter(
            torch.tensor(pr_params["betaSlow"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["betaSlow_trainable"],
        )
        self.betaSlow_scaleFac = nn.Parameter(
            torch.tensor(pr_params["betaSlow_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.hillcoef = nn.Parameter(
            torch.tensor(pr_params["hillcoef"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["hillcoef_trainable"],
        )
        self.hillcoef_scaleFac = nn.Parameter(
            torch.tensor(pr_params["hillcoef_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.hillaffinity = nn.Parameter(
            torch.tensor(pr_params["hillaffinity"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["hillaffinity_trainable"],
        )
        self.hillaffinity_scaleFac = nn.Parameter(
            torch.tensor(pr_params["hillaffinity_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.gamma = nn.Parameter(
            torch.tensor(pr_params["gamma"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["gamma_trainable"],
        )
        self.gamma_scaleFac = nn.Parameter(
            torch.tensor(pr_params["gamma_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )

        self.gdark = nn.Parameter(
            torch.tensor(pr_params["gdark"], dtype=dtype).expand(1, units),
            requires_grad=pr_params["gdark_trainable"],
        )
        self.gdark_scaleFac = nn.Parameter(
            torch.tensor(pr_params["gdark_scaleFac"], dtype=dtype).expand(1, units),
            requires_grad=False,
        )
        self.timeBin = nn.Parameter(torch.tensor(pr_params["timeBin"], dtype=dtype), requires_grad=True)

    def rieke_model(
        self,
        X_fun,
        TimeStep,
        sigma,
        phi,
        eta,
        cgmp2cur,
        cgmphill,
        cdark,
        beta,
        betaSlow,
        hillcoef,
        hillaffinity,
        gamma,
        gdark,
    ):
        darkCurrent = gdark**cgmphill * cgmp2cur / 2
        gdark = (2 * darkCurrent / cgmp2cur) ** (1 / cgmphill)

        cur2ca = beta * cdark / darkCurrent
        smax = eta / phi * gdark * (1 + (cdark / hillaffinity) ** hillcoef)

        num_points = X_fun.shape[1]

        g_prev = gdark + X_fun[:, 0, :] * 0
        s_prev = gdark * eta / phi + X_fun[:, 0, :] * 0
        c_prev = cdark + X_fun[:, 0, :] * 0
        r_prev = X_fun[:, 0, :] * gamma / sigma
        p_prev = (eta + r_prev) / phi

        g = torch.zeros(
            (X_fun.shape[0], num_points, X_fun.shape[2]),
            dtype=X_fun.dtype,
            device=X_fun.device,
        )
        g[:, 0, :] = X_fun[:, 0, :] * 0

        for pnt in range(1, num_points):
            r_curr = r_prev + TimeStep * (-sigma * r_prev)
            r_curr = r_curr + gamma * X_fun[:, pnt - 1, :]
            p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
            c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev**cgmphill) / 2 - beta * c_prev)
            s_curr = smax / (1 + (c_curr / hillaffinity) ** hillcoef)
            g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

            g[:, pnt, :] = g_curr

            g_prev = g_curr
            s_prev = s_curr
            c_prev = c_curr
            p_prev = p_curr
            r_prev = r_curr

        outputs = -(cgmp2cur * g**cgmphill) / 2

        return outputs

    def forward(self, inputs):
        X_fun = inputs

        timeBin = self.timeBin
        # frameTime = timeBin  # ms
        # upSamp_fac = int(frameTime / timeBin)
        TimeStep = 1e-3 * timeBin

        # if upSamp_fac > 1:
        #     X_fun = X_fun.repeat_interleave(upSamp_fac, dim=1)
        #     X_fun = X_fun / upSamp_fac  # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma * self.gamma_scaleFac) / timeBin
        gdark = self.gdark * self.gdark_scaleFac

        outputs = []

        for pr_type in range(self.units):
            outputs.append(
                self.rieke_model(
                    X_fun[:, pr_type, ...],
                    TimeStep,
                    sigma[:, pr_type],
                    phi[:, pr_type],
                    eta[:, pr_type],
                    cgmp2cur[:, pr_type],
                    cgmphill[:, pr_type],
                    cdark[:, pr_type],
                    beta[:, pr_type],
                    betaSlow[:, pr_type],
                    hillcoef[:, pr_type],
                    hillaffinity[:, pr_type],
                    gamma[:, pr_type],
                    gdark[:, pr_type],
                )
            )
        outputs = torch.stack(outputs, dim=1)

        # outputs = self.rieke_model(
        #     X_fun,
        #     TimeStep,
        #     sigma,
        #     phi,
        #     eta,
        #     cgmp2cur,
        #     cgmphill,
        #     cdark,
        #     beta,
        #     betaSlow,
        #     hillcoef,
        #     hillaffinity,
        #     gamma,
        #     gdark,
        # )

        # if upSamp_fac > 1:
        #     outputs = outputs[:, upSamp_fac - 1 :: upSamp_fac, :]

        return outputs
