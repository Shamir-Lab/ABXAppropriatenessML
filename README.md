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

Our data-specific parser generates 6 main pandas dataframes:
* **`Patient_df`** Contains demographics (static features).
* **`Drug_df`** Drugs entries.
* **`Culture_df`** Culture entries.
* **`Culture_antibiotic_df`** Cultures paired with antibiotic checked on it.
* **`Procedures_df`** Procedures entries.
* **`Labs_df`** Lab tests and vital signs (longitudinal features).

#### Patient dataframe format (Patient_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
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
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
| subject_id         | int            |
| hadm_id            | int            |
| itemid             | int            |
| hours_from_charttime_to_target_time | float64 |

#### Culture dataframe format (Culture_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
| subject_id         | int            |
| hadm_id            | int            |
| spec_type_desc     | String         |
| org_name           | String         |
| hours_from_charttime_to_target_time | float64 |

#### Culture Antibiotics dataframe format (Culture_antibiotic_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
| subject_id         | int            |
| hadm_id            | int            |
| org_name           | String         |
| ab_name            | String         |
| hours_from_charttime_to_target_time | float64 |
| interpretation     | String         |

#### Procedures dataframe format (Procedures_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
| subject_id         | int            |
| hadm_id            | int            |
| hours_from_charttime_to_target_time | float64 |
| label              | String         |

#### Lab and Vital signs dataframe format (Lab_df):
|   Columns           | Data type      | 
|---------------------|----------------|
| identifier         | object         |
| subject_id         | int            |
| hadm_id            | int            |
| itemid             | int            |
| label              | String         |
| valuenum           | float64        |
| hours_from_charttime_to_target_time | float64 |


