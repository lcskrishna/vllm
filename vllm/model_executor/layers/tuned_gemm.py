import torch
import torch.nn.functional as F
from rocsolidxgemm import rocb_create_extension,rocb_mm
from hipbsolidxgemm import hipb_create_extension,hipb_mm
import os
import yaml
import pandas as pd
from vllm import custom_ops


class TunedGemm:
    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
    def load_best_sols(self):
        tune_file = os.environ.get('VLLM_PERF_CSV')
        if tune_file is not None:
            self.bestsols = pd.read_csv(tune_file,index_col=[0])
            print(self.bestsols)
    def apply_custom(self,ds):
        M,N,K = ds['M'],ds['N'],ds['K']
        #apply custom matvec (only for f16 dtype)
        if N==1:
            ds1 = ds.copy()
            ds1['libtype'] = 'custom'
            if K==8192 and (M==1280 or M==7168):
                ds1['solidx'] = 8
                return ds1
            elif K==3584 and M==8192:
                ds1['solidx'] = 8
                return ds1
            elif K<=8192 and K%8==0 and M%4==0:
                ds1['solidx'] = 1
                return ds1
        return ds
    def create_ds(self):
        df = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = self.apply_custom(df.iloc[i])
            key = (ds['M'],ds['N'],ds['K'])
            if ds['libtype']=='hipblaslt': soltype = 1
            elif ds['libtype']=='rocblas': soltype = 2
            elif ds['libtype']=='custom': soltype = 3
            solds[key] = (soltype,int(ds['solidx']))
        self.solids = solds
        #print('>>>',solds)
    def query_sol(self,m,n,k):
        return self.solids.get((m,n,k),(0,0))
    
    def mm(self,inp,weights):
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        soltype,solidx = self.query_sol(m=weights.shape[0],n=inp.shape[0],k=inp.shape[1])
        # print(soltype, solidx)
        if soltype==1:
            out = hipb_mm(inp,weights.t(),solidx)
        elif soltype==2:
            out = rocb_mm(inp,weights.t(),solidx)
        else:
            #print('>>>Tgemm Default',inp.shape,weights.shape,soltype,solidx)
            out = F.linear(inp,weights)
        return out

tgemm = TunedGemm()
