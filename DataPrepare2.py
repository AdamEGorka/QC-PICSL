import pandas as pd

# Load the CSV files and build a datafram
imageqc_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MAYOADIRL_MRI_IMAGEQC_12_08_15_23May2023.csv",
                         dtype={14: str})
quality_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MAYOADIRL_MRI_QUALITY_ADNI3_23May2023.csv")
qc_t1_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\QC_ADNI_T1_ASHS_20230627_allQCdata.csv")
#mri_list_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MRI3TListWithNIFTIPath_10172022.tsv", sep="\t")
mri_list_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\ADNIMERGE_master_20221231.csv", low_memory=False)
print(mri_list_df.head())

# Just doing this for now
merged_df = pd.merge(qc_t1_df, mri_list_df, left_on=['SCANDATE', 'ID'], right_on=['EXAMDATE_MERGE', 'PTID'], how='inner')
print(merged_df['MTL_RIGHT'].head())

merged_df['SCANDATE'] = merged_df['SCANDATE'].astype(str) #Awful but its easier for later

#combined_df_adni = pd.merge(quality_df, quality_df, left_on='SCANDATE', right_on='EXAMDATE_MERGE', how='inner')

def get_data_df():
    return merged_df
