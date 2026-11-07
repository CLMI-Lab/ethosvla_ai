"""
EthosVLA: Cross-Dataset Master Aggregator
Aggregates and compares results across ALFRED/SHIELD, RLBench, and 3D Mask Datasets
"""

import os
import pandas as pd

def aggregate_all_datasets():
    print("=" * 80)
    print(" ETHOSVLA: CROSS-DATASET PUBLICATION BENCHMARK AGGREGATION")
    print(" Datasets: ALFRED (Household), RLBench (Manipulation), 3D Mask (Spatial Volumetric)")
    print("=" * 80)
    
    # Paths to the 3 dataset tables
    alfred_path = 'Code shammo/Baseline/Table_5_Final.csv'
    rlbench_path = 'Code shammo/RLBench/Table_RLBench_Final.csv'
    mask_path = 'Code shammo/3D_Mask/Table_3D_Mask_Final.csv'
    
    df_alfred = pd.read_csv(alfred_path)
    df_rlbench = pd.read_csv(rlbench_path)
    df_mask = pd.read_csv(mask_path)
    
    # Merge on Model
    merged = pd.merge(df_alfred[['Model', 'HAR↑', 'TSR↑', 'PVS↓', 'BDI↓']], 
                      df_rlbench[['Model', 'HAR↑', 'TSR↑', 'PVS↓', 'BDI↓']], 
                      on='Model', suffixes=('_ALFRED', '_RLBench'))
                      
    merged = pd.merge(merged, 
                      df_mask[['Model', 'HAR↑', 'TSR↑', 'PVS↓', 'BDI↓']], 
                      on='Model')
    merged.rename(columns={'HAR↑': 'HAR↑_3DMask', 'TSR↑': 'TSR↑_3DMask', 'PVS↓': 'PVS↓_3DMask', 'BDI↓': 'BDI↓_3DMask'}, inplace=True)
    
    # Compute Cross-Dataset Averages
    merged['Avg_HAR↑'] = ((merged['HAR↑_ALFRED'] + merged['HAR↑_RLBench'] + merged['HAR↑_3DMask']) / 3).round(1)
    merged['Avg_TSR↑'] = ((merged['TSR↑_ALFRED'] + merged['TSR↑_RLBench'] + merged['TSR↑_3DMask']) / 3).round(1)
    merged['Avg_PVS↓'] = ((merged['PVS↓_ALFRED'] + merged['PVS↓_RLBench'] + merged['PVS↓_3DMask']) / 3).round(2)
    merged['Avg_BDI↓'] = ((merged['BDI↓_ALFRED'] + merged['BDI↓_RLBench'] + merged['BDI↓_3DMask']) / 3).round(2)
    
    out_csv = 'Code shammo/Table_Cross_Dataset_Final.csv'
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved cross-dataset master table to {out_csv}")
    print(merged[['Model', 'Avg_HAR↑', 'Avg_TSR↑', 'Avg_PVS↓', 'Avg_BDI↓']].to_string(index=False))
    return merged

if __name__ == '__main__':
    aggregate_all_datasets()
