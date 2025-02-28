import os 
import time 
import yaml 
from argparse import ArgumentParser

def commonOptions():

    parser = ArgumentParser()
    parser.add_argument("--files",type=str, default="nAOD_step_ttgamma_0009_run0/NAOD-00000_61544.root", help="List of files to process");
    #parser.add_argument("--files",type=str, default="/cms/cephfs/data/store/user/apiccine/FullProduction/FullR2/UL17/Round2/Batch1/postLHE_step/v241121/nAOD_step_ttgamma_0009_run0/NAOD-00000_61544.root", help="List of files to process");
    parser.add_argument("--reload",  action='store_true', default=False, help="Force conversion of hdf to pytorch")
    parser.add_argument("--device", type=str, default='cpu', help="Which device (cpu, gpu index) to use ");
    parser.add_argument("--name", type=str, default="", help="Name to store net.");
    parser.add_argument("--out_path", type=str, default=".", help="Name to store net.");
    parser.add_argument("--features", type=str, default="Lep1_pt,Lep2_pt,Lep1_eta,Lep2_eta,Lep2_phi,Lep2_phi,Jet1_pt,Jet1_eta,Jet1_phi,Jet2_pt,Jet2_eta,Jet2_phi,met_pt", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--forceRebuild", action="store_true", default=False, help="Force reproduction of torch tensors from rootfiles")
    parser.add_argument("--norm", action="store_true", default=False, help="Force reproduction of torch tensors from rootfiles")
    parser.add_argument("--configuration-file", type=str, default=None, help="Load parameters from toml configuration file. The configuration file will be overriden by other command line options. The --name argument will always be taken from the command line option and the default")
    return parser


def parse(parser):

    args = parser.parse_args()

    # A bit of juggling with the configuration so we store it in yml files
    if args.configuration_file:
        with open(args.configuration_file) as f:
            config = yaml.safe_load(f.read())
    else:
        config = {}
    config = {**config, **vars(args)}

    with open(f"config.yml","w") as f:
        f.write( yaml.dump(config)) 

    for op, val in config.items():
        setattr( args, op, val)

    return args

def handleOptions():

    parser = commonOptions()
    parser.add_argument("--epochs",type=int, default=100, help="Number of epochs to train the net");
    parser.add_argument("--batch-size",type=int, default=64, help="Minibatch size");
    parser.add_argument("--learning-rate", type=float, default=0.00000005, help="Optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--factor", type=float, default=0.1, help="Factor for lr scheduler")
    parser.add_argument("--patience", type=float, default=10, help="Number of epochs required for lr adjustment")

    return parse(parser)

def rocOptions():
    parser = commonOptions()
    parser.add_argument("--parametric", type=str, default=None, 
                        help="yaml file containing information needed for parametric likeliood")
    parser.add_argument("--dedicated",  type=str, default=None, 
                        help="yaml file containing information needed for dedicated likeliood")
    
    
    return parse(parser)
