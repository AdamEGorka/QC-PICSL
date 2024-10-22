import pandas as pd

# Load the CSV files
imageqc_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MAYOADIRL_MRI_IMAGEQC_12_08_15_23May2023.csv",
                         dtype={14: str})
quality_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MAYOADIRL_MRI_QUALITY_ADNI3_23May2023.csv")
qc_t1_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\QC_ADNI_T1_ASHS_20230627_allQCdata.csv")
#mri_list_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\MRI3TListWithNIFTIPath_10172022.tsv", sep="\t")
mri_list_df = pd.read_csv(r"\Users\adame\Desktop\QC Project\ADNIMERGE_master_20221231.csv", low_memory=False)
print(mri_list_df.head())

quality_df.columns = quality_df.columns.str.capitalize()
qc_t1_df.columns = qc_t1_df.columns.str.capitalize()
combined_df_adni = pd.merge(quality_df, qc_t1_df, how='inner')

print("combined",combined_df_adni.head())

# Drop the first character from the "LONI_IMAGE" column in imageqc_df and quality_df

# ask later
# use command line to go through images in itksnap loop
# Am i able to drop rows without a loni_image/UID?
# What is the format of dates?
# tmux
# tmux attach -t 3

imageqc_df = imageqc_df.dropna(subset=["loni_image"])  # drop rows without a loni_image

# Cast UID to int
imageqc_df["loni_image"] = imageqc_df["loni_image"].str[1:].astype(int)
quality_df["LONI_IMAGE"] = quality_df["LONI_IMAGE"].astype(int)

# Cast .tsv UID to int
mri_list_df = mri_list_df.dropna(
    subset=["IMAGEUID_bl"])  # Drop Null types since they cant be used and wont cast to ints
mri_list_df['IMAGEUID_bl'] = mri_list_df['IMAGEUID_bl'].astype(int)


# now need to select only the matching UID in column “LONI_IMAGE” to the column “IMAGEUID_T1” in the tsv
imageqc_df = pd.merge(imageqc_df, mri_list_df, left_on='loni_image', right_on='IMAGEUID_bl',
                      how='inner')
quality_df = pd.merge(quality_df, mri_list_df, left_on='LONI_IMAGE', right_on='IMAGEUID_bl',
                      how='inner')
# So at this point imageqc_df and quality_df only has rows where the UID matches with the .tsv
print(quality_df.head())
# Now merge the imageqc_df with qc_t1_df
imageqc_df.columns = imageqc_df.columns.str.upper()

# Convert the "date" column in imageqc_df to a datetime format

imageqc_df["series_date"] = pd.to_datetime(imageqc_df["SERIES_DATE"], format="%Y%m%d")
# print(imageqc_df.head())


# Convert the date column in qc_t1_df to a datetime format for easier merging
qc_t1_df["SCANDATE"] = pd.to_datetime(qc_t1_df["SCANDATE"], infer_datetime_format=True)

# should be merged now
merged_df = pd.merge(imageqc_df, qc_t1_df, left_on=["RID", "series_date"], right_on=["RID", "SCANDATE"], how="inner")
print(merged_df.head())


def get_data_df():
    return merged_df
