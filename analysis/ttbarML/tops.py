import awkward as ak
import uproot
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import IterativeExecutor, Runner
#from distributed import Client

from coffea.processor.accumulator import AccumulatorABC

class MyAccumulator(AccumulatorABC):
    def __init__(self, value=0):
        self.value = value

    def identity(self):
        """Return a new instance with an identity value (zero for sums)."""
        return MyAccumulator(0)

    def add(self, other):
        """Define how to merge another accumulator of the same type."""
        if not isinstance(other, MyAccumulator):
            raise ValueError("Cannot add different types of accumulators")
        self.value = np.append([self.value, other.value])
        return self


class TopQuarkRecoProcessor(processor.ProcessorABC):
    def __init__(self, output_file="top_quark_reco_events.root"):
        self.output_file = output_file
        self._accumulator = MyAccumulator()

    def process(self, events):
        dataset_name = events.metadata["dataset"]
        # Select GenParticles: Find top quarks (PDG ID ±6)
        tops = events.GenPart[(abs(events.GenPart.pdgId) == 6)]
        tops = tops[ak.argsort(tops.pt, axis=1, ascending=False)]  # Sort by pT
        has_top = ak.num(tops) > 0  # Require at least one top quark

        # Ensure top1 is positive and top2 is negative
        tops = ak.concatenate([ak.firsts(tops[tops.pdgId==6]), ak.firsts(tops[tops.pdgId==-6])])
        top1 = tops[tops.pdgId==6]
        top2 = tops[tops.pdgId==-6]

        # Select reconstructed leptons (electrons and muons)
        leptons = ak.concatenate([events.Electron, events.Muon], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]  # Sort by pT

        # Find pairs with opposite charge
        lep_pairs = ak.combinations(leptons, 2, axis=1, fields=["lep1", "lep2"])
        opp_charge = lep_pairs.lep1.charge * lep_pairs.lep2.charge < 0  # Opposite sign

        # Select first valid opposite-charge pair
        valid_pairs = lep_pairs[opp_charge]
        has_valid_pairs = ak.num(valid_pairs) > 0
        first_pair = ak.firsts(valid_pairs)  # Returns None for empty arrays

        # Ensure lepton1 is positive and lepton2 is negative
        pos_first = first_pair.lep1.charge > 0
        lepton1 = ak.where(pos_first, first_pair.lep1, first_pair.lep2)  # Positive charge first
        lepton2 = ak.where(pos_first, first_pair.lep2, first_pair.lep1)  # Negative charge second
        #lepton1 = lepton1[(lepton1.pt>25) & (np.abs(lepton1.eta<2.4))]
        #lepton2 = lepton2[(lepton2.pt>25) & (np.abs(lepton2.eta<2.4))]

        # Handle cases where no valid lepton pair exists
        #lepton1 = ak.fill_none(lepton1, None)
        #lepton2 = ak.fill_none(lepton2, None)
        #lepton1 = ak.fill_none(lepton1, ak.Array([]))
        #lepton2 = ak.fill_none(lepton2, ak.Array([]))

        # Select reconstructed jets (highest pT, first 4)
        jets = events.Jet
        jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        #jets = jets[(jets.pt > 30) & (np.abs(jets.eta) < 2.4)]
        desired_jets = 8
        jets = ak.pad_none(jets, desired_jets, clip=True)  # Keep up to 4 jets

        # Apply event selection (only keep events with at least one top quark)
        #has_top = has_top & ~ak.is_none(~ak.pad_none(lepton1, 1)) & ~ak.is_none(~ak.pad_none(lepton2, 1)) & ~ak.is_none(~ak.pad_none(jets, 1))
        has_top = has_top & ~ak.is_none(lepton1) & ~ak.is_none(lepton2) & ~ak.is_none(lepton1.pt>25) & ~ak.is_none(lepton2.pt>15)# & ~ak.is_none(np.abs(lepton1.eta) < 2.4) & ~ak.is_none(np.abs(lepton2.eta) < 2.4) & ~ak.any(ak.is_none(jets))# & ak.num(jets)>=4
        #has_top = ak.fill_none(has_top, False)
        selected_events = events[has_top]
        selected_tops = tops[has_top]
        selected_top1 = top1[has_top]
        selected_top2 = top2[has_top]
        selected_lepton1 = lepton1[has_top]
        selected_lepton2 = lepton2[has_top]
        selected_jets = ak.fill_none(jets[has_top][:,0:desired_jets], 0)
        print(f'{selected_lepton1=}')
        print(f'{ak.is_none(lepton1)=}')
        print(f'{selected_lepton1.pt=}', '\n\n\n')
        print(f'{selected_tops.pt=}', '\n\n\n')
        print(f'{selected_jets.pt=}')

        # Convert selected data to a dictionary for ROOT output
        output_data = {
            #"event": ak.to_numpy(selected_events.event),
            #"run": ak.to_numpy(selected_events.run),
            #"luminosityBlock": ak.to_numpy(selected_events.luminosityBlock),

            # Top quarks
            "ttbar_px": ak.to_numpy((selected_top1+selected_top2).px),
            "ttbar_py": ak.to_numpy((selected_top1+selected_top2).py),
            "ttbar_pz": ak.to_numpy((selected_top1+selected_top2).pz),
            "ttbar_pt": ak.to_numpy((selected_top1+selected_top2).pt),
            "ttbar_eta": ak.to_numpy((selected_top1+selected_top2).eta),
            "ttbar_phi": ak.to_numpy((selected_top1+selected_top2).phi),
            "ttbar_mass": ak.to_numpy((selected_top1+selected_top2).mass),
            "ttbar_energy": ak.to_numpy((selected_top1+selected_top2).energy),

            "top1_px": ak.to_numpy(np.cos(selected_top1.phi) * selected_top1.pt),
            "top1_py": ak.to_numpy(np.sin(selected_top1.phi) * selected_top1.pt),
            "top1_pz": ak.to_numpy(np.sinh(selected_top1.eta) * selected_top1.pt),
            "top1_pt": ak.to_numpy(selected_top1.pt),
            "top1_eta": ak.to_numpy(selected_top1.eta),
            "top1_phi": ak.to_numpy(selected_top1.phi),
            "top1_mass": ak.to_numpy(selected_top1.mass),
            "top1_energy": ak.to_numpy(selected_top1.energy),
            "top1_pdgId": ak.to_numpy(selected_top1.pdgId),

            "top2_px": ak.to_numpy(np.cos(selected_top2.phi) * selected_top2.pt),
            "top2_py": ak.to_numpy(np.sin(selected_top2.phi) * selected_top2.pt),
            "top2_pz": ak.to_numpy(np.sinh(selected_top2.eta) * selected_top2.pt),
            "top2_pt": ak.to_numpy(selected_top2.pt),
            "top2_eta": ak.to_numpy(selected_top2.eta),
            "top2_phi": ak.to_numpy(selected_top2.phi),
            "top2_mass": ak.to_numpy(selected_top2.mass),
            "top2_energy": ak.to_numpy(selected_top2.energy),
            "top2_pdgId": ak.to_numpy(selected_top2.pdgId),

            # Leptons (positive-charge first, negative-charge second)
            "lepton1_px": ak.to_numpy(np.cos(selected_lepton1.phi) * selected_lepton1.pt),
            "lepton1_py": ak.to_numpy(np.sin(selected_lepton1.phi) * selected_lepton1.pt),
            "lepton1_pz": ak.to_numpy(np.sinh(selected_lepton1.eta) * selected_lepton1.pt),
            "lepton1_pt": ak.to_numpy(selected_lepton1.pt),
            "lepton1_eta": ak.to_numpy(selected_lepton1.eta),
            "lepton1_phi": ak.to_numpy(selected_lepton1.phi),
            "lepton1_mass": ak.to_numpy(selected_lepton1.mass),
            "lepton1_energy": ak.to_numpy(np.cosh(selected_lepton1.eta) * np.sqrt(np.square(selected_lepton1.pt) + np.square(selected_lepton2.mass))),
            "lepton1_charge": ak.to_numpy(selected_lepton1.charge),
            "lepton1_pdgId": ak.to_numpy(selected_lepton1.pdgId),

            "lepton2_px": ak.to_numpy(np.cos(selected_lepton2.phi) * selected_lepton2.pt),
            "lepton2_py": ak.to_numpy(np.sin(selected_lepton2.phi) * selected_lepton2.pt),
            "lepton2_pz": ak.to_numpy(np.sinh(selected_lepton2.eta) * selected_lepton2.pt),
            "lepton2_pt": ak.to_numpy(selected_lepton2.pt),
            "lepton2_eta": ak.to_numpy(selected_lepton2.eta),
            "lepton2_phi": ak.to_numpy(selected_lepton2.phi),
            "lepton2_mass": ak.to_numpy(selected_lepton2.mass),
            "lepton2_energy": ak.to_numpy(np.cosh(selected_lepton2.eta) * np.sqrt(np.square(selected_lepton2.pt) + np.square(selected_lepton2.mass))),
            "lepton2_charge": ak.to_numpy(selected_lepton2.charge),
            "lepton2_pdgId": ak.to_numpy(selected_lepton2.pdgId),


            # Jets
            #"jet1_px": ak.to_numpy(selected_jets.px[:,0]),
            #"jet1_py": ak.to_numpy(selected_jets.py[:,0]),
            #"jet1_pz": ak.to_numpy(selected_jets.pz[:,0]),
            #"jet1_pt": ak.to_numpy(selected_jets.pt[:,0]),
            #"jet1_eta": ak.to_numpy(selected_jets.eta[:,0]),
            #"jet1_phi": ak.to_numpy(selected_jets.phi[:,0]),
            #"jet1_mass": ak.to_numpy(selected_jets.mass[:,0]),
            #"jet1_energy": ak.to_numpy(selected_jets.energy[:,0]),

            #"jet2_px": ak.to_numpy(selected_jets.px[:,1]),
            #"jet2_py": ak.to_numpy(selected_jets.py[:,1]),
            #"jet2_pz": ak.to_numpy(selected_jets.pz[:,1]),
            #"jet2_pt": ak.to_numpy(selected_jets.pt[:,1]),
            #"jet2_eta": ak.to_numpy(selected_jets.eta[:,1]),
            #"jet2_phi": ak.to_numpy(selected_jets.phi[:,1]),
            #"jet2_mass": ak.to_numpy(selected_jets.mass[:,1]),
            #"jet2_energy": ak.to_numpy(selected_jets.energy[:,1]),

            #"jet3_px": ak.to_numpy(selected_jets.px[:,2]),
            #"jet3_py": ak.to_numpy(selected_jets.py[:,2]),
            #"jet3_pz": ak.to_numpy(selected_jets.pz[:,2]),
            #"jet3_pt": ak.to_numpy(selected_jets.pt[:,2]),
            #"jet3_eta": ak.to_numpy(selected_jets.eta[:,2]),
            #"jet3_phi": ak.to_numpy(selected_jets.phi[:,2]),
            #"jet3_mass": ak.to_numpy(selected_jets.mass[:,2]),
            #"jet3_energy": ak.to_numpy(selected_jets.energy[:,2]),

            #"jet4_px": ak.to_numpy(selected_jets.px[:,3]),
            #"jet4_py": ak.to_numpy(selected_jets.py[:,3]),
            #"jet4_pz": ak.to_numpy(selected_jets.pz[:,3]),
            #"jet4_pt": ak.to_numpy(selected_jets.pt[:,3]),
            #"jet4_eta": ak.to_numpy(selected_jets.eta[:,3]),
            #"jet4_phi": ak.to_numpy(selected_jets.phi[:,3]),
            #"jet4_mass": ak.to_numpy(selected_jets.mass[:,3]),
            #"jet4_energy": ak.to_numpy(selected_jets.energy[:,3]),

            ##"jet_px": ak.to_numpy(ak.flatten(selected_jets.px)),
            ##"jet_py": ak.to_numpy(ak.flatten(selected_jets.py)),
            ##"jet_pz": ak.to_numpy(ak.flatten(selected_jets.pz)),
            ##"jet_pt": ak.to_numpy(ak.flatten(selected_jets.pt)),
            ##"jet_eta": ak.to_numpy(ak.flatten(selected_jets.eta)),
            ##"jet_phi": ak.to_numpy(ak.flatten(selected_jets.phi)),
            ##"jet_mass": ak.to_numpy(ak.fill_none(selected_jets.mass, 0)),
            #"jet_mass": ak.to_numpy(ak.flatten(ak.fill_none(selected_jets.mass, 0))),
            ##"jet_mass": ak.to_numpy(ak.flatten(selected_jets.mass)),


            # MET
            "met_px": ak.to_numpy(events.MET[has_top].px),
            "met_py": ak.to_numpy(events.MET[has_top].py),
        }
        for i in range(desired_jets):
            output_data[f"jet{i+1}_px"] = ak.to_numpy(selected_jets.px[:,i])
            output_data[f"jet{i+1}_py"] = ak.to_numpy(selected_jets.py[:,i])
            output_data[f"jet{i+1}_pz"] = ak.to_numpy(selected_jets.pz[:,i])
            output_data[f"jet{i+1}_pt"] = ak.to_numpy(selected_jets.pt[:,i])
            output_data[f"jet{i+1}_eta"] = ak.to_numpy(selected_jets.eta[:,i])
            output_data[f"jet{i+1}_phi"] = ak.to_numpy(selected_jets.phi[:,i])
            #output_data[f"jet{i+1}_mass"] = ak.to_numpy(ak.fill_none(selected_jets.mass, 0[:,i])
            output_data[f"jet{i+1}_energy"] = ak.to_numpy(selected_jets.energy[:,i])

        print(output_data)
        #return {'Events': output_data}
        output_file = f"top_quark_reco_events_{dataset_name}.root"
        with uproot.recreate(output_file) as root_file:
            root_file['Events'] = output_data
        return {}

    @property
    def accumulator(self):
        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator

# Function to run processor over multiple files
def run_processor(file_list, dataset_name="TopQuarkDataset", output_file="top_quark_reco_events.root"):
    fileset = {dataset_name: file_list}
    fileset = {f[:-5]:[f] for f in file_list}
    print(fileset)

    executor = IterativeExecutor()  # Use multi-threading for efficiency
    runner = Runner(executor=executor, schema=NanoAODSchema)

    output = runner(fileset, treename="Events", processor_instance=TopQuarkRecoProcessor())

    # Save merged output to a ROOT file
    with uproot.recreate(output_file) as root_file:
        '''
        '''
        evt = {}
        for ky, array in output.items():
            print(ky)
            for key, array in array:
                if key in evt:
                    for k,v in evt[k].items():
                        evt[key][k] = np.append([v,array])
                else:
                    evt[key] = array
                    print(key, array)
                #root_file[key] = {key: array}
        root_file["Events"] = evt
        #root_file["Events"] = output['Events']

    print(f"Saved top quark, opposite-sign lepton pairs, and jet events to {output_file}")

if __name__ == "__main__":
    # List of multiple NanoAOD files
    nanoaod_files = [
        "NAOD-00000_898.root", "NAOD-00000_919.root", "NAOD-00000_959.root", "NAOD-00000_996.root", "NAOD-00000_899.root", "NAOD-00000_944.root", "NAOD-00000_976.root", "NAOD-00000_997.root", "NAOD-00000_916.root", "NAOD-00000_945.root", "NAOD-00000_983.root", "NAOD-00000_917.root", "NAOD-00000_946.root", "NAOD-00000_984.root"
        #"NAOD-00000_898.root"
    ]

    run_processor(nanoaod_files)

