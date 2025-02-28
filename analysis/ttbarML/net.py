import torch.nn as nn
import torch 

cost =  nn.MSELoss(reduction='mean')#reduction='sum')

class Net(nn.Module):
    def __init__(self, features, feature_division, device):
        super().__init__()
        initial_neurons = 4
        self.main_module = nn.Sequential( 
            #nn.Linear(features, 32),
            #nn.ReLU(True),
            #nn.Linear(32, 16),
            #nn.ReLU(True),
            #nn.Linear(16, 8),
            #nn.ReLU(True),
            #nn.Linear(8, 1),
            #nn.Sigmoid(),
            nn.Linear(features-feature_division, initial_neurons),#32),  # Input layer with 27 inputs and 64 neurons
            #nn.Sigmoid(),          # Activation function
            nn.LeakyReLU(0.2),
            #nn.ReLU(),          # Activation function
            nn.Linear(initial_neurons, initial_neurons//2),#32, 16),  # Hidden layer with 32 neurons
            #nn.Sigmoid(),          # Activation function
            nn.LeakyReLU(0.2),
            #nn.ReLU(),
            nn.Linear(initial_neurons//2, feature_division) #16, feature_division)    # Output layer with 3 outputs (for the target values
        )
        self.main_module.to(device)
        self.main_module.type(torch.float64)


    def forward(self, x):
        return self.main_module(x)
            

class Model:
    def __init__(self, features, feature_division, device):
        '''
        features: inputs used to train the neural network
        device: device used to train the neural network
        '''
        self.net  = Net(features, feature_division, device=device)
        #self.bkg  = torch.tensor([0], device=device, dtype=torch.float64)
        #self.sgnl = torch.tensor([1], device=device, dtype=torch.float64)

    def cost_from_batch(self, targets, features, device ):
        '''
        features: input features to the neural network
        weight_sm: weights of the background events
        weight_bsm: weights of the signal events
        sm_mean: mean of the background weights
        bsm_mean: mean of the signal weights
        '''
        half_length = features.size(0) // 2
        #print(self.bkg.expand(1, half_length).shape, self.sgnl.expand(1, features.size(0) - half_length).shape)

        #cost.weight = torch.ones_like(features.T[0])
        #targets     = features[:,0:3]
        #print(len(features[0]))
        #print(features[0:2])
        #print('input', features.shape)
        #print('output', targets.shape)
        #print('input', features)
        #print('output', targets)
        #print('here')
        #print('model', self.net(features))
        #print('model', self.net(features).shape)
        #targets     = features.T[:,3]
        #print(f'{self.net(features)=} {targets=}')
        #print(f'{self.net(features).shape=} {targets.shape=}')
        #targets     = torch.cat([self.bkg.expand(1, half_length), self.sgnl.expand(1, features.size(0) - half_length)], axis=1).ravel()
        
        #print(self.net(features).detach().cpu().numpy(), targets.detach().cpu().numpy())
        pred = self.net(features)
        # prediction: pt = 0, eta = 1, phi = 2, pz = 3, mass = 4, energy = 5
        # targets:    pt = 0, eta = 1, phi = 2, pz = 3, mass = 4, energy = 5
                                #0          1          2         3           4           5
        #features = np.array([ttbar.pt, ttbar.eta, ttbar.phi, ttbar.pz, ttbar.mass, ttbar.energy, Lep1.pt, Lep2.pt, Lep1.eta, Lep2.eta, Lep2.phi, Lep2.phi, Jet1.pt, Jet1.eta, Jet1.phi, Jet2.pt, Jet2.eta, Jet2.phi, Jet3.pt, Jet3.eta, Jet3.phi, Jet4.pt, Jet4.eta, Jet4.phi, met.pt, PT.pt]).T
        #                                     pT diff                                                          DeltaR                                                    systemat mass  ttbar mass                 system E    ttbar E                      pT system^2 + pZ system^2                                 ttbar pT^2 + ttbar pZ^2
        #return torch.mean(torch.square(pred[:,0] - targets[:,0])) + torch.mean(torch.square(pred[:,1] - targets[:,1]) + torch.square(pred[:,2] - targets[:,2])) + torch.mean(pred[:,4] - targets[:,4]) + torch.mean(pred[:,5] - targets[:,5]) + torch.mean((torch.square(pred[:,0]) + torch.square(pred[:,3])) - (torch.square(pred[:,0]) + torch.square(pred[:,3])))
        #return torch.mean(torch.square(pred[:,0] - targets[:,0])) + torch.mean(torch.square(pred[:,1] - targets[:,1]) + torch.square(pred[:,2] - targets[:,2]))# + torch.mean(torch.square(pred[:,3] - targets[:,3]))
        #print(f'In model net={self.net(features)[0]} targerts={targets[:][0].detach().cpu().numpy()} cost={cost(self.net(features), targets).detach().cpu().numpy()}')
        #print(f'In model net={self.net(features)[0]} targerts={targets[:][0].detach().cpu().numpy()} cost={torch.mean(torch.square(self.net(features)[:,0] - targets[:,0])).detach().cpu().numpy()}')
        #return torch.mean(torch.square(self.net(features)[:,0] - targets[:,0]))
        return torch.mean(torch.square(self.net(features)[:,0] - targets[:,0]))
        #return cost(self.net(features), targets)
        #return cost(self.net(features).ravel(), targets)
