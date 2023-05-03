import argparse, sys, os
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from sklearn import svm

## parse command-line arguments
# process arguments
parser = argparse.ArgumentParser(description='Drug response prediction on new drugs')

parser.add_argument('-i', '--input', required=True, help='path to input drug response prediction results (csv file)')
parser.add_argument('-smiles', '--input_smiles', required=True, help='path to input SMILES encoding of molecules (csv file)')
parser.add_argument('-o', '--output', default='./', help='path to output directory, default=\'./\'')
parser.add_argument('-m', '--model', default='PRISM', help='which model was used in the previous drug response prediction process, default model: PRISM.')


args = parser.parse_args()

# check input
if args.model == "PRISM":
    input_file = os.path.join(args.input, "PRISM_prediction.csv")
elif args.model == "GDSC":
    input_file = os.path.join(args.input, "IC50_prediction.csv")

if not os.path.exists(args.input):
    sys.exit('The input path does not exist.')
if not os.path.exists(input_file):
    sys.exit('The input file does not exist.')
if input_file[-4:] != ".csv":
    sys.exit('The input file is not a csv file.')


if not os.path.exists(args.input_smiles):
    sys.exit('The input path does not exist.')
if args.input_smiles[-4:] != ".csv":
    sys.exit('The input file is not a csv file.')


# check output
if not os.path.isdir(args.output):
    sys.exit('The output directory does not exist.')


class new_drug_prediction:
    def __init__(self):
        self.load_prediction()
        self.load_input_mol()
        self.prepare_fingerprints()
        self.sensitivity_prediction()
        self.compute_output()
        
    def load_prediction(self):
        if args.model == "PRISM":
            self.cadrres_pred = pd.read_csv(os.path.join(args.input, "PRISM_prediction.csv"), index_col=0, header=[0,1])
            mapping_df = pd.read_csv('/scDrug/data/PRISM_drugID_smiles_map.csv', index_col = 0)
            self.train_smiles = mapping_df.loc[self.cadrres_pred.columns.get_level_values(0), 'smiles'].values
        elif args.model == "GDSC":
            self.cadrres_pred = pd.read_csv(os.path.join(args.input, "IC50_prediction.csv"), index_col=0, header=[0,1])
            mapping_df = pd.read_csv('/scDrug/data/GDSC_drugID_smiles_map.csv', index_col = 0)
            self.cadrres_pred.columns = self.cadrres_pred.columns.droplevel(0)
            avai_mol = list(set(self.cadrres_pred.columns) & set(mapping_df.index))
            self.cadrres_pred = self.cadrres_pred[avai_mol].T.reset_index().groupby("Drug Name", as_index=False).mean().set_index("Drug Name").T
            self.train_smiles = mapping_df.loc[avai_mol , 'smiles'].values
        else:
            print('invalid model name.')
    
    def RDKfp_convert(self, smiles_ls):
        mol_rdkit = list(map(Chem.MolFromSmiles,smiles_ls))
        fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in mol_rdkit]
        fps = np.array(fps)
        return fps
    
    def load_input_mol(self):
        self.input_smiles_df = pd.read_csv(args.input_smiles, index_col=1)
        self.input_smiles = self.input_smiles_df.index.tolist()

    def prepare_fingerprints(self):
        self.X = self.RDKfp_convert(self.train_smiles)
        self.input_mol_fps = self.RDKfp_convert(self.input_smiles)
    
    def sensitivity_prediction(self):
        pred_auc = np.empty((len(self.input_smiles), len(self.cadrres_pred.index)))
        pred_auc[:] = np.nan
        self.pred_auc_df = pd.DataFrame(pred_auc, columns= self.cadrres_pred.index.tolist(), index = self.input_smiles)

        for cluster in self.cadrres_pred.index:
            y = self.cadrres_pred.loc[cluster]
            model = svm.SVR()
            model.fit(self.X, y)
            cluster_pred = model.predict(self.input_mol_fps)
            self.pred_auc_df[cluster] = cluster_pred
        print("saving drug response prediction of new drug.")
        self.pred_auc_df.to_csv(os.path.join(args.output, "new_drug_prediction.csv"), index = True)
    
    def compute_output(self):
        if args.model == "PRISM":
            self.pred_auc_df = 1-self.pred_auc_df
            self.pred_auc_output = self.pred_auc_df.reset_index(names='smiles').melt(id_vars=["smiles"], var_name = 'cluster', 
                                                        value_vars= self.pred_auc_df.columns.tolist(), 
                                                        value_name = "AUC prediction")
            #self.pred_auc_output['classification'] = ['potnetial' if pred < 0.4 else ('inactive' if pred > 0.8 else 'unclear') 
            #                                        for pred in 1-self.pred_auc_output['AUC prediction']]
            self.pred_auc_output['rank'] = list(map(int, self.pred_auc_output.groupby("cluster")["AUC prediction"].rank(ascending = True)))
            self.pred_auc_output= self.pred_auc_output.sort_values(['cluster','rank'])
            self.pred_auc_output['mol_name'] = self.input_smiles_df.loc[self.pred_auc_output['smiles']]['mol_name'].values

            print("saving drug level prediction")
            self.pred_auc_output = self.pred_auc_output[['cluster', 'rank', 'mol_name', 'smiles', 'AUC prediction']]
            self.pred_auc_output = self.pred_auc_output.set_index(["cluster"])
            self.pred_auc_output.to_csv(os.path.join(args.output, "drug_rank_prediction.csv"), index = True)

        elif args.model == "GDSC":
            self.pred_ic50_output = self.pred_auc_df.reset_index(names='smiles').melt(id_vars=['smiles'], var_name = 'cluster', 
                                                                    value_vars=self.pred_auc_df.columns.tolist(), 
                                                                    value_name = "IC50 prediction")
            self.pred_ic50_output['rank'] = list(map(int, self.pred_ic50_output.groupby("cluster")["IC50 prediction"].rank(ascending = True)))
            self.pred_ic50_output= self.pred_ic50_output.sort_values(['cluster','rank'])
            self.pred_ic50_output['mol_name'] = self.input_smiles_df.loc[self.pred_ic50_output['smiles']]['mol_name'].values

            print("saving drug level prediction")
            self.pred_ic50_output = self.pred_ic50_output[['cluster', 'rank', 'mol_name', 'smiles', 'IC50 prediction']]
            self.pred_ic50_output.to_csv(os.path.join(args.output, "drug_rank_prediction.csv"), index = False)
        else:
            print('invalid model name.')

job = new_drug_prediction()