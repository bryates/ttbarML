from torch.utils.data import DataLoader
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import torch.utils.data as data
import numpy as np 
import awkward as ak

import uproot
import torch
import glob
import tqdm
import os


class eftDataLoader( data.Dataset ):
    def __init__(self, args):

        #self.files   = [args['files']]
        self.files   = glob.glob(args['files'])
        #print(self.files)
        self.dtype   = np.float64
        
        self.feature_list  = args['features'].split(',')
        #print(self.feature_list)
        self.out_path = args['out_path']
        self.forceRebuild = args['forceRebuild']
        self.device = args['device']

        self.build_tensors()
        self.load_tensors()

    def __len__( self ):
        return self.features.shape[0]


    def __getitem__(self, idx):
        return self.features[idx,:]

                
    def build_tensors( self ):

        '''
        Labeling the data
        '''
        self.name = "tops"

        if self.forceRebuild:
            os.system(f'rm -f {self.out_path}/*.p')

        '''
        Load PyTorch files
        '''
        redoFeatures = not os.path.isfile(f'{self.out_path}/features.p')

        outputs={}
        if redoFeatures:
            '''
            Reset
            '''
            print("Will redo tensor with input features")
            outputs['features'        ] = np.empty( shape=(0, len(self.feature_list)), dtype=self.dtype)

        if not redoFeatures:
            return 

        print("Loading files, this may take a while")
        for fil in tqdm.tqdm(self.files):
            #print(fil)
            '''
            Open root file in uproot and load `Events`
            This is where I should load the p4 for leps, jets, MET, and gen tops
            '''
            #tf = uproot.open( fil )
            events = NanoEventsFactory.from_root(fil, schemaclass=NanoAODSchema).events()[:10]
            #print(len(events))
            #events = tf["Events"]

            if redoFeatures:
                tops = events.GenPart
                tops = tops[np.abs(tops.pdgId)==6]
                ttbar_mask = (ak.num(events.Electron) + ak.num(events.Muon) == 2) & (ak.num(events.Jet) >= 4) & (ak.num(tops) >= 2)# & (ak.all(np.abs(tops.eta)<4))
                tops = events.GenPart[ttbar_mask]
                tops = tops[np.abs(tops.pdgId)==6]
                ttbar = tops[np.abs(tops.pdgId)==6][:,0:2].sum()
                #ttbar = ak.with_nametops[np.abs(tops.pdgId)==6][:,0:2].sum(), 'PtEtaPhiMCandidate')
                evts = {}
                evts['top1'] = tops[:,0]
                evts['ttbar'] = ttbar
                #ttbar = ttbar & (np.abs(tops.sum().eta)<4)
                #tops = events.GenPart[ttbar]
                el = events.Electron[ttbar_mask]
                mu = events.Muon[ttbar_mask]
                g_el = events.GenPart[ttbar_mask]
                g_el = g_el[np.abs(g_el.pdgId)==11]
                g_mu = events.GenPart[ttbar_mask]
                g_mu = g_mu[np.abs(g_mu.pdgId)==13]
                jet = events.Jet[ttbar_mask]
                Leps = ak.with_name(ak.concatenate([el, mu], axis=1), 'PtEtaPhiMCandidate')
                g_Leps = ak.with_name(ak.concatenate([g_el, g_mu], axis=1), 'PtEtaPhiMCandidate')
                #print(g_Leps.pdgId)
                #Lep1 = ak.with_name(ak.concatenate([el, mu], axis=1), 'PtEtaPhiMCandidate')[:, 0]
                Lep1 = ak.pad_none(Leps[Leps.pdgId>0], 1)[:,0]
                evts['Lep1'] = Lep1
                g_Lep1 = ak.pad_none(g_Leps, 1)[:,0]
                evts['gLep1'] = g_Lep1
                #Lep1 = ak.fill_none(ak.pad_none(ak.with_name(ak.concatenate([el, mu], axis=1), 'PtEtaPhiMCandidate'), 2), -1)[:, 0]
                #Lep2 = ak.with_name(ak.concatenate([el[:,el.pdgId<0], mu[:,mu.pdgId<0]], axis=1), 'PtEtaPhiMCandidate')[:, 1]
                Lep2 = ak.pad_none(Leps[Leps.pdgId<0], 1)[:,0]
                evts['Lep2'] = Lep2
                hasLeps = ~ak.is_none(Lep1.pdgId) & ~ak.is_none(Lep2.pdgId) # Only 2l OS
                hasLeps = hasLeps & ~ak.is_none(g_Lep1.pdgId)
                Jet1 = jet[:,0]
                Jet2 = jet[:,1]
                Jet3 = jet[:,2]
                Jet4 = jet[:,3]
                evts['Jet1'] = Jet1
                evts['Jet2'] = Jet2
                evts['Jet3'] = Jet3
                evts['Jet4'] = Jet4
                met = events.MET[ttbar_mask]
                evts['met'] = met
                PT = (Lep1 + Lep2 + Jet1 + Jet2 + Jet3 + Jet4 + met)
                evts['PT'] = PT
                #features = np.array([Lep1.pt, Lep1.eta, Lep1.phi, Lep2.pt, Lep2.eta, Lep2.phi, Jet1.pt, Jet1.eta, Jet1.phi, Jet2.pt, Jet2.eta, Jet2.phi, met.pt]).T
                #print(f'{ttbar.pt=}')
                #print(f'{ttbar.eta=}')
                #print(f'{ttbar.phi=}')
                #print(*ttbar.pt[0:10])
                #print(f'{Lep1.pt=}')
                #print(f'{PT.pt=}')
                #features = np.array([tops.x, tops.y, tops.z, Lep1.x, Lep1.y, Lep1.z, Lep2.x, Lep2.y, Lep2.z, Jet1.x, Jet1.y, Jet1.z, Jet2.x, Jet2.y, Jet2.z, Jet3.x, Jet3.y, Jet3.z, Jet4.x, Jet4.y, Jet4.z, met.x]).T
                #features = np.array([ttbar.pt, ttbar.eta, ttbar.phi, Lep1.pt, Lep1.eta, Lep1.phi, Lep2.pt, Lep2.eta, Lep2.phi, Jet1.pt, Jet1.eta, Jet1.phi, Jet2.pt, Jet2.eta, Jet2.phi, Jet3.pt, Jet3.eta, Jet3.phi, Jet4.pt, Jet4.eta, Jet4.phi, met.pt, PT.pt]).T
                #features = np.array([ttbar.pt, ttbar.eta, ttbar.phi, ttbar.pz, ttbar.mass, ttbar.energy, Lep1.pt, Lep2.pt, Lep1.eta, Lep2.eta, Lep2.phi, Lep2.phi, Jet1.pt, Jet1.eta, Jet1.phi, Jet2.pt, Jet2.eta, Jet2.phi, Jet3.pt, Jet3.eta, Jet3.phi, Jet4.pt, Jet4.eta, Jet4.phi, met.pt, PT.pt]).T
                feat = []
                for feature in self.feature_list:
                    obj,var = feature.strip().split('_')
                    # TODO automate this! (evts[obj][var] doesn't work)
                    if 'pt' in var:
                        feat.append(evts[obj].pt[hasLeps])
                    elif 'pz' in var:
                        feat.append(evts[obj].pz[hasLeps])
                    elif 'eta' in var:
                        feat.append(evts[obj].eta[hasLeps])
                    elif 'phi' in var:
                        feat.append(evts[obj].phi[hasLeps])
                    elif 'mass' in var:
                        feat.append(evts[obj].mass[hasLeps])
                    elif 'energy' in var:
                        feat.append(evts[obj].energy[hasLeps])
                    else:
                        raise Exception(f'No rule for {var}!')
                #print(features)
                #print(len(feat))
                #print(f"{outputs['features'].T}")
                #print(len(outputs['features'].T), len(features.T))
                #print(f'{np.array(feat).T=}')
                outputs['features'] = np.append( outputs['features'], np.array(feat).T, axis=0)
                #outputs['features'] = np.append( outputs['features'], features, axis=0)
            #outputs['features'] = np.array( [[10., 10.], [10, 10], [10, 9], [1, 1], [1, 2]] )
            #print(f'Loaded first ttbar_pt {outputs["features"][0][0]}')
            #print(f'Loaded first ttbar_pt {outputs["features"]}')
            #print(f'Loaded first ttbar_pt {outputs["features"].shape}')
            #print(f'Loaded first ttbar_pt {type(outputs["features"])}')
            #break # for development

        # writing tensors to file
        for output in outputs:
            t = torch.from_numpy( outputs[output] )
            torch.save( t, f'{self.out_path}/{output}.p')

    def load_tensors(self):
        self.features   = torch.load( f'{self.out_path}/features.p').to(device = self.device)
