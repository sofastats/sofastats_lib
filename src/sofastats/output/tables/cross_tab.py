"""
Under top-level can only have chains not trees.

GOOD:
age > gender
country > browser > gender  <====== chain

BAD:
age > gender   <======= tree
    > car
country > browser
        > car
        > age

Useful to be able to look at different things by one thing
e.g. for each country (rows) age break down, browser breakdown, and car breakdown (sic) ;-)

But that's enough complexity. Anything more, better making multiple, individually clear tables.
"""
from dataclasses import dataclass
from functools import partial
from itertools import product

import pandas as pd

from sofastats.output.interfaces import (
    DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY, HTMLItemSpec, OutputItemType, CommonDesign)
from sofastats.output.styles.utils import get_style_spec
from sofastats.output.tables.interfaces import BLANK, Column, Metric, PctType, Row
from sofastats.output.tables.utils.html_fixes import (
    fix_top_left_box, merge_cols_of_blanks, merge_rows_of_blanks)
from sofastats.output.tables.utils.misc import (apply_index_styles, get_data_from_spec,
                                                get_df_pre_pivot_with_pcts, get_raw_df, set_table_styles)
from sofastats.output.tables.utils.multi_index_sort import (
    get_order_rules_for_multi_index_branches, get_sorted_multi_index_list)
from sofastats.utils.misc import get_pandas_friendly_name, correct_str_dps
from sofastats.utils.pandas import no_silent_downcasting, infer_objects_no_copy

pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1_000)

from collections.abc import Collection

def get_all_metrics_df_from_vars(data, *, row_vars: list[str], col_vars: list[str],
        n_row_fillers: int = 0, n_col_fillers: int = 0, pct_metrics: Collection[Metric], decimal_points: int = 2,
        debug=False) -> pd.DataFrame:
    """
    Includes at least the Freq metric but potentially the percentage ones as well.

    Start with creating a dataframe with a column for each row variables design,
    then one for each column variables design, then freq. All in a fixed order we can rely on.
    Which is why we can build the columns from the input variable names, in order.
    We know the last column is the count so we add 'n' as the final column. E.g. Country, Gender, Age Group, n.

            Country  Gender Age Group    n
    0            NZ  Female     20-29    9
    1            NZ  Female     30-39    7
    ...
    70          USA   TOTAL     TOTAL  133
    71        TOTAL   TOTAL     TOTAL  349

    Now to add the variable names themselves e.g.

            Country  Gender Age Group    n country_var gender_var age_group_var
    0            NZ  Female     20-29    9     Country     Gender     Age Group
    1            NZ  Female     30-39    7     Country     Gender     Age Group
    ...
    70          USA   TOTAL     TOTAL  133     Country     Gender     Age Group
    71        TOTAL   TOTAL     TOTAL  349     Country     Gender     Age Group

    Note - the source, un-pivoted, df has all TOTAL values calculated and identified in the val columns already.

    OK, so now we have a proper df. Time to add in any row or column filler columns
    (some will pivot to rows and others to columns in the final df)
    __BLANK__

            Country  Gender Age Group    n country_var gender_var age_group_var col_filler_var_0 col_filler_0 metric
    0            NZ  Female     20-29    9     Country     Gender     Age Group        __blank__    __blank__   Freq
    1            NZ  Female     30-39    7     Country     Gender     Age Group        __blank__    __blank__   Freq
    ...
    70          USA   TOTAL     TOTAL  133     Country     Gender     Age Group        __blank__    __blank__   Freq
    71        TOTAL   TOTAL     TOTAL  349     Country     Gender     Age Group        __blank__    __blank__   Freq
    ...

    Then pivot the data (at this stage, simple so we have a required input to make more df_pre_pivots
    for any row or col pcts data):

    age_group_var                             Age Group
    Age Group                                     20-29     30-39     40-64       65+      < 20     TOTAL
    col_filler_var_0                          __blank__ __blank__ __blank__ __blank__ __blank__ __blank__
    col_filler_0                              __blank__ __blank__ __blank__ __blank__ __blank__ __blank__
    metric                                         Freq      Freq      Freq      Freq      Freq      Freq
    country_var Country     gender_var Gender
    Country     NZ          Gender     Female         9         7        21        26        16        79
                                       Male          11        11        15        16        12        65
                                       TOTAL         20        18        36        42        28       144
    ...

    Then we generate additional df_pre_pivots for row pcts and col pcts as appropriate. And we pivot the final df.

       country_var Country gender_var  Gender web_browser_var Web Browser age_group_var Age Group metric      n
    0      Country      NZ     Gender  Female     Web Browser      Chrome     Age Group     20-29  Row %  17.65
    0      Country      NZ     Gender  Female     Web Browser      Chrome     Age Group     30-39  Row %   5.88
    ...
    0      Country     USA     Gender   TOTAL     Web Browser       TOTAL     Age Group     TOTAL  Row %  100.0

    AND (note - metric and row cols in different positions depending on whether Row or Col)

       web_browser_var Web Browser age_group_var Age Group metric country_var      Country gender_var  Gender      n
    0      Web Browser      Chrome     Age Group     20-29  Col %     Country           NZ     Gender  Female   60.0
    0      Web Browser      Chrome     Age Group     20-29  Col %     Country           NZ     Gender    Male   40.0
    ...
    0      Web Browser       TOTAL     Age Group     TOTAL  Col %     Country          USA     Gender   TOTAL  100.0

    Then we pivot the new, combined df_pre_pivot and metric splays across intoFreq, Row %, and Col % as appropriate.
    Here is a case with Freq, Row %, and Col %:

    Note - yet to have the columns reordered so we have Freq Row % Freq Row % etc

    web_browser_var                           Web Browser                                                                                                                                                                             ...
    Web Browser                                    Chrome                         Firefox                           TOTAL                          Chrome   Firefox     TOTAL    Chrome                               Firefox         ...     TOTAL                         Chrome   Firefox     TOTAL    Chrome                               Firefox                                 TOTAL                                Chrome   Firefox     TOTAL
    age_group_var                               Age Group                       Age Group                       Age Group                       Age Group Age Group Age Group Age Group                             Age Group         ... Age Group                      Age Group Age Group Age Group Age Group                             Age Group                             Age Group                             Age Group Age Group Age Group
    Age Group                                       20-29 30-39 40-64  65+ < 20     20-29 30-39 40-64  65+ < 20     20-29 30-39 40-64  65+ < 20     TOTAL     TOTAL     TOTAL     20-29  30-39  40-64    65+   < 20     20-29  30-39  ...     30-39  40-64    65+   < 20     TOTAL     TOTAL     TOTAL     20-29  30-39  40-64    65+   < 20     20-29  30-39  40-64    65+   < 20     20-29  30-39  40-64    65+   < 20     TOTAL     TOTAL     TOTAL
    metric                                           Freq  Freq  Freq Freq Freq      Freq  Freq  Freq Freq Freq      Freq  Freq  Freq Freq Freq      Freq      Freq      Freq     Row %  Row %  Row %  Row %  Row %     Row %  Row %  ...     Row %  Row %  Row %  Row %     Row %     Row %     Row %     Col %  Col %  Col %  Col %  Col %     Col %  Col %  Col %  Col %  Col %     Col %  Col %  Col %  Col %  Col %     Col %     Col %     Col %
    country_var Country     gender_var Gender                                                                                                                                                                                         ...
    Country     NZ          Gender     Female           3     1     3    8    2         3     1     8    9    6         6     2    11   17    8        17        27        44     17.65   5.88  17.65  47.06  11.76     11.11    3.7  ...      4.55   25.0  38.64  18.18     100.0     100.0     100.0      60.0   50.0   75.0  66.67  33.33      37.5   20.0  61.54  56.25   60.0     46.15  28.57  64.71  60.71   50.0     58.62     51.92     54.32
                                       Male             2     1     1    4    4         5     4     5    7    4         7     5     6   11    8        12        25        37     16.67   8.33   8.33  33.33  33.33      20.0   16.0  ...     13.51  16.22  29.73  21.62     100.0     100.0     100.0      40.0   50.0   25.0  33.33  66.67      62.5   80.0  38.46  43.75   40.0     53.85  71.43  35.29  39.29   50.0     41.38     48.08     45.68
                                       TOTAL            5     2     4   12    6         8     5    13   16   10        13     7    17   28   16        29        52        81     17.24    6.9  13.79  41.38  20.69     15.38   9.62  ...      8.64  20.99  34.57  19.75     100.0     100.0     100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0     100.0     100.0
                South Korea Gender     Female           3     2     2    0    2         3     2     2    3    2         6     4     4    3    4         9        12        21     33.33  22.22  22.22    0.0  22.22      25.0  16.67  ...     19.05  19.05  14.29  19.05     100.0     100.0     100.0      60.0   40.0   50.0    0.0   40.0      75.0  66.67  33.33   37.5  66.67     66.67   50.0   40.0  27.27   50.0     40.91      50.0     45.65
                                       Male             2     3     2    3    3         1     1     4    5    1         3     4     6    8    4        13        12        25     15.38  23.08  15.38  23.08  23.08      8.33   8.33  ...      16.0   24.0   32.0   16.0     100.0     100.0     100.0      40.0   60.0   50.0  100.0   60.0      25.0  33.33  66.67   62.5  33.33     33.33   50.0   60.0  72.73   50.0     59.09      50.0     54.35
                                       TOTAL            5     5     4    3    5         4     3     6    8    3         9     8    10   11    8        22        24        46     22.73  22.73  18.18  13.64  22.73     16.67   12.5  ...     17.39  21.74  23.91  17.39     100.0     100.0     100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0     100.0     100.0
                TOTAL       Gender     Female           7     5     8   16    4        10     4    19   18    8        17     9    27   34   12        40        59        99      17.5   12.5   20.0   40.0   10.0     16.95   6.78  ...      9.09  27.27  34.34  12.12     100.0     100.0     100.0     43.75  38.46  42.11  48.48  33.33     47.62   40.0  63.33  51.43  53.33     45.95  39.13   55.1   50.0  44.44     43.01     53.15     48.53
                                       Male             9     8    11   17    8        11     6    11   17    7        20    14    22   34   15        53        52       105     16.98  15.09  20.75  32.08  15.09     21.15  11.54  ...     13.33  20.95  32.38  14.29     100.0     100.0     100.0     56.25  61.54  57.89  51.52  66.67     52.38   60.0  36.67  48.57  46.67     54.05  60.87   44.9   50.0  55.56     56.99     46.85     51.47
                                       TOTAL           16    13    19   33   12        21    10    30   35   15        37    23    49   68   27        93       111       204      17.2  13.98  20.43  35.48   12.9     18.92   9.01  ...     11.27  24.02  33.33  13.24     100.0     100.0     100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0     100.0     100.0
                USA         Gender     Female           1     2     3    8    0         4     1     9    6    0         5     3    12   14    0        14        20        34      7.14  14.29  21.43  57.14    0.0      20.0    5.0  ...      8.82  35.29  41.18    0.0     100.0     100.0     100.0     16.67  33.33  27.27  44.44    0.0     44.44   50.0  81.82  54.55    0.0     33.33   37.5  54.55  48.28    0.0     33.33     57.14     44.16
                                       Male             5     4     8   10    1         5     1     2    5    2        10     5    10   15    3        28        15        43     17.86  14.29  28.57  35.71   3.57     33.33   6.67  ...     11.63  23.26  34.88   6.98     100.0     100.0     100.0     83.33  66.67  72.73  55.56  100.0     55.56   50.0  18.18  45.45  100.0     66.67   62.5  45.45  51.72  100.0     66.67     42.86     55.84
                                       TOTAL            6     6    11   18    1         9     2    11   11    2        15     8    22   29    3        42        35        77     14.29  14.29  26.19  42.86   2.38     25.71   5.71  ...     10.39  28.57  37.66    3.9     100.0     100.0     100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0  100.0     100.0     100.0     100.0

    Finally, round numbers
    """
    ## Need country_var = Country and country_val = NZ etc. - we start with the vals (which is what the data contains)
    orig_data_columns = row_vars + col_vars + ['n', ]
    df_pre_pivot = pd.DataFrame(data, columns=orig_data_columns)
    all_variables = row_vars + col_vars
    index_cols = []
    column_cols = []
    for var in all_variables:
        """       columns already in df_pre_pivot because taken directly from source data
                     |       |        |        |
                     V       V        V        V
                 Country  Gender  Age Group    n  country_var gender_var age_group_var
        0            NZ   Female      20-29    9      Country     Gender     Age Group
        1            NZ   Female      30-39    7      Country     Gender     Age Group
        ...
        70          USA    TOTAL      TOTAL  133      Country     Gender     Age Group
        71        TOTAL    TOTAL      TOTAL  349      Country     Gender     Age Group
        """
        ## populate var and val cells e.g. 'Age Group' and '< 20'
        df_pre_pivot[get_pandas_friendly_name(var, '_var')] = var  ## e.g. country_var = Country
        cols2add = [get_pandas_friendly_name(var, '_var'), var]  ## e.g. [country_var, Country, ]
        if var in row_vars:
            index_cols.extend(cols2add)
        elif var in col_vars:
            column_cols.extend(cols2add)
        else:
            raise Exception(f"{var=} not found in either {row_vars=} or {col_vars=}")
    ## only add what is needed to fill gaps
    for i in range(n_row_fillers):
        df_pre_pivot[f'row_filler_var_{i}'] = BLANK
        df_pre_pivot[f'row_filler_{i}'] = BLANK
        index_cols.extend([f'row_filler_var_{i}', f'row_filler_{i}'])
    for i in range(n_col_fillers):
        df_pre_pivot[f'col_filler_var_{i}'] = BLANK
        df_pre_pivot[f'col_filler_{i}'] = BLANK
        column_cols.extend([f'col_filler_var_{i}', f'col_filler_{i}'])
    column_cols.append('metric')
    df_pre_pivot['metric'] = 'Freq'
    df_pre_pivot['n'] = df_pre_pivot['n'].astype(pd.Int64Dtype())
    if debug: print(df_pre_pivot)
    df_pre_pivots = [df_pre_pivot, ]
    df = df_pre_pivot.pivot(index=index_cols, columns=column_cols, values='n')  ## missing rows e.g. if we have no rows for females < 20 in the USA, now appear as NAs so we need to fill them in df
    with no_silent_downcasting():
        df = infer_objects_no_copy(df.fillna(0))  ## needed so we can round values (can't round a NA). Also need to do later because of gaps appearing when pivoted then too
    if pct_metrics:
        if Metric.ROW_PCT in pct_metrics:
            df_pre_pivot_inc_row_pct = get_df_pre_pivot_with_pcts(
                df, pct_type=PctType.ROW_PCT, decimal_points=decimal_points, debug=debug)
            df_pre_pivots.append(df_pre_pivot_inc_row_pct)
        if Metric.COL_PCT in pct_metrics:
            df_pre_pivot_inc_col_pct = get_df_pre_pivot_with_pcts(
                df, pct_type=PctType.COL_PCT, decimal_points=decimal_points, debug=debug)
            df_pre_pivots.append(df_pre_pivot_inc_col_pct)
    df_pre_pivot = pd.concat(df_pre_pivots)
    df = df_pre_pivot.pivot(index=index_cols, columns=column_cols, values='n')
    with no_silent_downcasting():
        df = infer_objects_no_copy(df.fillna(0))
    df = df.astype(str)
    ## have to ensure all significant digits are showing e.g. 3.33 and 1.0 or 0.0 won't align nicely
    correct_string_dps = partial(correct_str_dps, decimal_points=decimal_points)
    df = df.map(correct_string_dps)
    return df


@dataclass(frozen=False, kw_only=True)
class CrossTabDesign(CommonDesign):
    """
    Args:
        row_variable_designs: list of Rows
        column_variable_designs: list of Columns
    """
    row_variable_designs: list[Row] = DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY
    column_variable_designs: list[Column] = DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY

    debug: bool = False
    verbose: bool = False

    @staticmethod
    def _get_dupes(_vars: Collection[str]) -> set[str]:
        dupes = set()
        seen = set()
        for var in _vars:
            if var in seen:
                dupes.add(var)
            else:
                seen.add(var)
        return dupes

    @property
    def totalled_vars(self) -> list[str]:
        tot_vars = []
        for row_spec in self.row_variable_designs:
            tot_vars.extend(row_spec.self_and_descendant_totalled_vars)
        for col_spec in self.column_variable_designs:
            tot_vars.extend(col_spec.self_and_descendant_totalled_vars)
        return tot_vars

    def _get_max_dim_depth(self, *, is_col=False) -> int:
        max_depth = 0
        dim_specs = self.column_variable_designs if is_col else self.row_variable_designs
        for dim_spec in dim_specs:
            dim_depth = len(dim_spec.self_and_descendant_vars)
            if dim_depth > max_depth:
                max_depth = dim_depth
        return max_depth

    @property
    def max_row_depth(self) -> int:
        return self._get_max_dim_depth()

    @property
    def max_col_depth(self) -> int:
        return self._get_max_dim_depth(is_col=True)

    def __post_init__(self):
        CommonDesign.__post_init__(self)
        row_dupes = CrossTabDesign._get_dupes([spec.variable_name for spec in self.row_variable_designs])
        if row_dupes:
            raise ValueError(f"Duplicate top-level variable(s) detected in row dimension - {sorted(row_dupes)}")
        col_dupes = CrossTabDesign._get_dupes([spec.variable_name for spec in self.column_variable_designs])
        if col_dupes:
            raise ValueError(f"Duplicate top-level variable(s) detected in column dimension - {sorted(col_dupes)}")
        ## var can't be in both row and col e.g. car vs country > car
        for row_spec, col_spec in product(self.row_variable_designs, self.column_variable_designs):
            row_spec_vars = set([row_spec.variable_name] + row_spec.descendant_vars)
            col_spec_vars = set([col_spec.variable_name] + col_spec.descendant_vars)
            overlapping_vars = row_spec_vars.intersection(col_spec_vars)
            if overlapping_vars:
                raise ValueError("Variables can't appear in both rows and columns. "
                    f"Found the following overlapping variable(s): {', '.join(overlapping_vars)}")

    def get_df_from_row_spec(self, cur, *, row_spec_idx: int) -> pd.DataFrame:
        """
        get a combined df for, e.g. the combined top df. Or the middle df. Or the bottom df. Or whatever you have.
        e.g.
        row_variables_design_1 = Row(variable='country', has_total=True,
            child=(variable='gender', has_total=True))
        vs
        column_variables_design_1 = Column(variable='Age Group', has_total=True)
        column_variables_design_2 = Column(variable='Web Browser', has_total=True,
            child=Column(variable='Age Group', has_total=True, pct_metrics=[Metric.ROW_PCT, Metric.COL_PCT]))
        column_variables_design_3 = Column(variable='Standard Age Group', has_total=True)
        """
        row_spec = self.row_variable_designs[row_spec_idx]
        row_vars = row_spec.self_and_descendant_vars
        n_row_fillers = self.max_row_depth - len(row_vars)
        df_cols = []
        for col_spec in self.column_variable_designs:
            col_vars = col_spec.self_and_descendant_vars
            totalled_variables = row_spec.self_and_descendant_totalled_vars + col_spec.self_and_descendant_totalled_vars
            all_variables = row_vars + col_vars
            data = get_data_from_spec(cur, dbe_spec=self.dbe_spec,
                source_table_name=self.source_table_name, table_filter_sql=self.table_filter_sql,
                all_variables=all_variables, totalled_variables=totalled_variables, debug=self.debug)
            df_col = get_all_metrics_df_from_vars(data, row_vars=row_vars, col_vars=col_vars,
                n_row_fillers=n_row_fillers, n_col_fillers=self.max_col_depth - len(col_vars),
                pct_metrics=col_spec.self_or_descendant_pct_metrics, decimal_points=self.decimal_points, debug=self.debug)
            df_cols.append(df_col)
        df = df_cols[0]
        df_cols_remaining = df_cols[1:]
        row_merge_on = []
        for row_var in row_vars:
            row_merge_on.append(get_pandas_friendly_name(row_var, '_var'))
            row_merge_on.append(row_var)
        for i in range(n_row_fillers):
            row_merge_on.append(f'row_filler_var_{i}')
            row_merge_on.append(f'row_filler_{i}')
        for df_next_col in df_cols_remaining:
            df = df.merge(df_next_col, how='outer', on=row_merge_on)
        return df

    def get_tbl_df(self, cur) -> pd.DataFrame:
        """
        For each row_variable_designs get a completed df and then merge those.

        Note - using pd.concat or df.merge(how='outer') has the same result, but I use merge for horizontal joining
        to avoid repeating the row dimension columns e.g. country and gender.

        Basically we are merging left and right dfs. Merging is typically on an id field that both parts share.
        In this case there are as many fields to merge on as there are fields in the row index -
        in this example there are 4 (var_00, val_00, var_01, and val_01).
        There is one added complexity because the column is multi-index.
        We need to supply a tuple with an item (possibly an empty string) for each level.
        In this case there are two levels (browser and age_group). So we merge on
        [('var_00', ''), ('val_00', ''), ('var_01', ''), ('val_01', '')]
        If there were three row levels and four col levels we would need something like:
        [('var_00', '', '', ''), ('val_00', '', '', ''), ... ('val_02', '', '', '')]

        BOTTOM LEFT:
        browser    var_00       val_00     var_01     val_01 Chrome                       Firefox
        agegroup                                                <20 20-29 30-39 40-64 65+     <20 20-29 30-39 40-64 65+
        0         Country           NZ  __blank__  __blank__     10    19    17    28  44      25    26    14    38  48
        ...

        BOTTOM RIGHT:
        agegroup   var_00       val_00     var_01     val_01 <20 20-29 30-39 40-64 65+
        dummy
        0         Country           NZ  __blank__  __blank__  35    45    31    66  92
        ...

        Note - we flatten out the row multi-index using reset_index().
        This flattening results in a column per row variable e.g. one for country and one for gender
         (at this point we're ignoring the labelling step where we split each row variable e.g. for country into Country (var) and NZ (val)).
        Given it is a column, it has to have as many levels as the column dimension columns.
        So if there are two column dimension levels each row column will need to be a two-tuple e.g. ('gender', '').
        If there were three column dimension levels the row column would need to be a three-tuple e.g. ('gender', '', '').
        """
        dfs = [self.get_df_from_row_spec(cur, row_spec_idx=row_spec_idx)
            for row_spec_idx in range(len(self.row_variable_designs))]
        ## COMBINE using pandas JOINing (the big magic trick at the middle of this approach to complex table-making)
        ## Unfortunately, delegating to Pandas means we can't fix anything intrinsic to what Pandas does.
        ## And there is a bug (from my point of view) whenever tables are merged with the same variables at the top level.
        ## To prevent this we have to disallow variable reuse at top-level.
        ## transpose, join, and re-transpose back. JOINing on rows works differently from columns and will include all items in sub-levels under the correct upper levels even if missing from the first multi-index
        ## E.g. if Age Group > 40-64 is missing from the first index it will not be appended on the end but will be alongside all its siblings so we end up with Age Group > >20, 20-29 30-39, 40-64, 65+
        ## Note - variable levels (odd numbered levels if 1 is the top level) should be in the same order as they were originally
        df_t = dfs[0].T
        dfs_remaining = dfs[1:]
        for df_next in dfs_remaining:
            df_t = df_t.join(df_next.T, how='outer')
        df = df_t.T  ## re-transpose back so cols are cols and rows are rows again
        if self.debug: print(f"\nCOMBINED:\n{df}")
        ## Sorting indexes
        raw_df = get_raw_df(cur, dbe_spec=self.dbe_spec, source_table_name=self.source_table_name, debug=self.debug)
        order_rules_for_row_multi_index_branches = get_order_rules_for_multi_index_branches(self.row_variable_designs)
        order_rules_for_col_multi_index_branches = get_order_rules_for_multi_index_branches(self.column_variable_designs)
        ## COLS
        unsorted_col_multi_index_list = list(df.columns)
        sorted_col_multi_index_list = get_sorted_multi_index_list(
            unsorted_col_multi_index_list, order_rules_for_multi_index_branches=order_rules_for_col_multi_index_branches,
            sort_orders=self.sort_orders, raw_df=raw_df, has_metrics=True, debug=self.debug)
        sorted_col_multi_index = pd.MultiIndex.from_tuples(sorted_col_multi_index_list)  ## https://pandas.pydata.org/docs/user_guide/advanced.html
        ## ROWS
        unsorted_row_multi_index_list = list(df.index)
        sorted_row_multi_index_list = get_sorted_multi_index_list(
            unsorted_row_multi_index_list, order_rules_for_multi_index_branches=order_rules_for_row_multi_index_branches,
            sort_orders=self.sort_orders, raw_df=raw_df, has_metrics=False, debug=self.debug)
        sorted_row_multi_index = pd.MultiIndex.from_tuples(sorted_row_multi_index_list)  ## https://pandas.pydata.org/docs/user_guide/advanced.html
        df = df.reindex(index=sorted_row_multi_index, columns=sorted_col_multi_index)
        if self.debug: print(f"\nORDERED:\n{df}")
        return df

    def to_html_design(self) -> HTMLItemSpec:
        get_tbl_df_for_cur = partial(self.get_tbl_df)
        df = get_tbl_df_for_cur(self.cur)
        pd_styler = set_table_styles(df.style)
        style_spec = get_style_spec(style_name=self.style_name)
        pd_styler = apply_index_styles(df, style_spec, pd_styler, axis='rows')
        pd_styler = apply_index_styles(df, style_spec, pd_styler, axis='columns')
        raw_tbl_html = pd_styler.to_html()
        if self.debug:
            print(raw_tbl_html)
        ## Fix
        html = raw_tbl_html
        html = fix_top_left_box(html, style_spec, debug=self.debug, verbose=self.verbose)
        html = merge_cols_of_blanks(html, debug=self.debug)
        html = merge_rows_of_blanks(html, debug=self.debug, verbose=self.verbose)
        if self.debug:
            print(pd_styler.uuid)
            print(html)
        return HTMLItemSpec(
            html_item_str=html,
            output_item_type=OutputItemType.MAIN_TABLE,
            output_title=self.output_title,
            design_name=self.__class__.__name__,
            style_name=self.style_name,
        )
