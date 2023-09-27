import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, b_values, Delta, delta, gradient_strength, nparams):
        super(Net, self).__init__()

        self.b_values = b_values
        self.Delta = Delta
        self.delta = delta
        self.gradient_strength = gradient_strength

        self.layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.layers.extend([nn.Linear(len(b_values), len(b_values)), nn.PReLU()])
        self.encoder = nn.Sequential(*self.layers, nn.Linear(len(b_values), nparams))
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        
        X = self.dropout(X)
        params = torch.nn.functional.softplus(self.encoder(X))

        # constrain parameters to biophysically-realistic ranges
        f_ic = torch.clamp(params[:,0].unsqueeze(1), min=0.001, max=0.999)
        f_ees = torch.clamp(params[:,1].unsqueeze(1), min=0.001, max=0.999)
        r = torch.clamp(params[:,2].unsqueeze(1), min=0.001, max=14.999)
        d_ees = torch.clamp(params[:,3].unsqueeze(1), min=0.5, max=3)
        
        # sphere GPD approximation
        SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035
        ]
        
        alpha = torch.FloatTensor(SPHERE_TRASCENDENTAL_ROOTS) / (r)
        alpha2 = alpha ** 2
        alpha2D = alpha2 * 2
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)

        gamma = 2.675987e2
        first_factor = -2*(gamma*self.gradient_strength)**2 / 2

        delta = self.delta.unsqueeze(0).unsqueeze(2)
        Delta = self.Delta.unsqueeze(0).unsqueeze(2)
        
        summands = (alpha ** (-4) / (alpha2 * (r.unsqueeze(2))**2 - 2) * (
                            2 * delta - (
                            2 +
                            torch.exp(-alpha2D * (Delta - delta)) -
                            2 * torch.exp(-alpha2D * delta) -
                            2 * torch.exp(-alpha2D * Delta) +
                            torch.exp(-alpha2D * (Delta + delta))
                        ) / (alpha2D)
                    )
                )
        
        xi = (1 - f_ic - f_ees) * ((np.sqrt(np.pi) * torch.erf(np.sqrt(self.b_values * 2))) /
                (2 * np.sqrt(self.b_values * 2))) # astrosticks compartment
        xii = f_ic * torch.exp(torch.FloatTensor(first_factor) * torch.sum(summands, 2)) # sphere compartment
        xiii = f_ees * torch.exp(-self.b_values * d_ees) # ball compartment       
        X = xi + xii + xiii

        return X, f_ic, f_ees, r, d_ees
