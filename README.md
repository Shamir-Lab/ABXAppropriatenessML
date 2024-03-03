## *Predicting appropriateness of antibiotic treatment among ICU patients with hospital acquired infection*
Public repository containing code for model predicting antibiotic treatment appropriateness, as described in the manuscript "Predicting appropriateness of antibiotic treatment among ICU patients with hospital acquired infection"

#### Authors:
Ella Goldschmidt*, Ella Rannon*, Daniel Bernstein, Asaf Wasserman, Dan Coster, Ron Shamir

#### Modules
The repository contains the following modules:
* **`data_processing`** Code for preprocessing the data, including time-series feature creation, imputation, outlier removal, etc.
* **`feature_engineering`** Code for the creation of the features used by the model.
* **`feature_selection`** Code for feature selection methods and filtration of correlated and redundant features.
* **`feature_stats`** Code for conducting statistical tests on the features.
* **`models`** Code for the BalancedDataEnsemble model and a generic class for ABX appropriateness prediction model.

### Data:
The data used in our study is from MIMIC-III. This section describes the data format used for the code.
The 'target' variable represents whether the antibiotic treatment administered to the patient was appropriate, given the antibiogram results of a specific culture sample. A value of 1 indicates inappropriate treatment, while a value of 0 signifies appropriate treatment. The determination of appropriateness is based on the sensitivity of the identified bacteria to the antibiotics administered to the patient.

Our data-specific parser generates 6 main pandas dataframes:
* **`Patient_df`** Contains demographics (static features).
* **`Drug_df`** Drugs entries.
* **`Culture_df`** Culture entries.
* **`Culture_antibiotic_df`** Cultures paired with antibiotic checked on it.
* **`Procedures_df`** Procedures entries.
* **`Labs_df`** Lab tests and vital signs (longitudinal features).

#### Patient dataframe format (Patient_df):
This dataset encompasses all demographic details of the patients, with each individual assigned two specific identifiers: 'subject_id', a unique identifier that remains constant throughout a patient's lifetime, and 'hadm_id', which is unique to each hospital stay of a patient. We introduced the 'identifier' column, which combines 'subject_id' and 'hadm_id' using a hyphen (i.e. 'subject_id-hadm_id'). Essential information about each patient, including age, weight, height, and ethnicity, is captured. The 'target' variable serves as the label indicating the appropriateness of the treatment, which is essential only during the model training phase. There is no requirement for this label when performing inference. Furthermore, we calculate the time elapsed from hospital admission to the target time (the time of the relevant culture sample), as well as the duration from ICU admission to the target time.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| target             | int            |
| gender             | String         |
| age                | int            |
| admission_weight             | float64        |
| admission_height             | float64        |
| hours_from_admittime_to_target_time | float64 |
| hours_from_icutime_to_target_time   | float64 |
| ethnicity          | String         |

#### Drug dataframe format (Drug_df):
All the drugs administered in the hospital were aggregated into 11 categories and for each category. Here 'itemid' is the id of a specific drug and we calculate the time from administration to target time. This table will be used in our model to calculate the total amount of drugs administered from each category and used as a feature for the model.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| itemid             | int            |
| hours_from_charttime_to_target_time | float64 |

#### Culture dataframe format (Culture_df):
The cultures table contains two key attributes that later will serve as features: the tissue in which the sample was taken from, labeled as 'spec_type_desc', and the name of the organism detected within, referred to as 'org_name'. Additionally, we compute the duration from the time the culture was taken to the target time.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| spec_type_desc     | String         |
| org_name           | String         |
| hours_from_charttime_to_target_time | float64 |

#### Culture Antibiotics dataframe format (Culture_antibiotic_df):
The table includes outcomes of cultures obtained from patients, specifically focusing on the antibiotics tested against each culture, denoted as 'ab_name', and whether the bacteria were sensitive (indicated by "S") or resistant ("R") to the antibiotic, as recorded under 'interpretation'. Additionally, we compute the duration from the time the culture was taken to the target time.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| org_name           | String         |
| ab_name            | String         |
| hours_from_charttime_to_target_time | float64 |
| interpretation     | String         |

#### Procedures dataframe format (Procedures_df):
For all the procedures conducted in the hospital, our code generates features for several invasive procedures, categorizing them into four types: Arterial Line, Catheter, Ventilation, and Tubes. Only procedures with labels falling under one of these four categories should be included in the table (see double_names_mimic.json). Additionally, we compute the duration from the time of the procedure to the target time.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| hours_from_charttime_to_target_time | float64 |
| label              | String         |

#### Lab and Vital signs dataframe format (Lab_df):
This table stores lab measurements and vital signs, where 'itemid' represents the hospital-assigned ID for a specific measurement. The 'label' refers to the name of the measurement conducted, and 'valuenum' is the recorded numerical value. Additionally, we compute the duration from the time of measurement to the target time.
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | String         |
| subject_id         | int            |
| hadm_id            | int            |
| itemid             | int            |
| label              | String         |
| valuenum           | float64        |
| hours_from_charttime_to_target_time | float64 |


