from dataclasses import dataclass
from functools import partial

import pandas as pd

from sofastats.output.interfaces import (
    DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY,
    CommonDesign,
    HTMLItemSpec,
    OutputItemType,
)
from sofastats.output.styles.utils import get_style_spec
from sofastats.output.tables.interfaces import BLANK, PctType, Row
from sofastats.output.tables.utils.html_fixes import (
    fix_top_left_box,
    merge_cols_of_blanks,
)
from sofastats.output.tables.utils.misc import (
    apply_index_styles,
    get_data_from_spec,
    get_df_pre_pivot_with_pcts,
    get_raw_df,
    set_table_styles,
)
from sofastats.output.tables.utils.multi_index_sort import (
    get_metric2order,
    get_order_rules_for_multi_index_branches,
    get_sorted_multi_index_list,
)
from sofastats.utils.misc import correct_str_dps, get_pandas_friendly_name
from sofastats.utils.pandas import infer_objects_no_copy, no_silent_downcasting


def get_all_metrics_df_from_vars(data, *, row_vars: list[str],
        n_row_fillers: int = 0, inc_col_pct=False, decimal_points: int = 2, debug=False) -> pd.DataFrame:
    """
    Includes at least the Freq metric but potentially the percentage ones as well.

    Start with a column for each row var, then one for each col var, then freq. All in a fixed order we can rely on.
    Which is why we can build the columns from the input variable names, with _val suffix, in order.
    We know the last column is the count so we add 'n' as the final column.
    E.g. country_val, gender_val, n.

            Country  Gender    n
    0            NZ  Female   44
    1            NZ    Male   37
    ...
    11        TOTAL   TOTAL  204
    ...
    Note - the source, un-pivoted df has all TOTAL values calculated and identified in the val columns already.

    OK, so now we have a proper df. Time to add extra columns e.g. alongside Country we add country_var with all values
    set to 'Country', and country as 'USA', 'South Korea', etc

      metric country_var      Country gender_var  Gender       n
    0  Col %     Country           NZ     Gender  Female  54.321
    0  Col %     Country           NZ     Gender    Male  45.679
    ...
    0  Col %     Country          USA     Gender   TOTAL   100.0

    Then add in any row or column filler columns (some will pivot to rows and others to columns in the final df)
    with __BLANK__ e.g. (if another config from that followed through in this example):
    ...

    Then pivot

    metric                                      Freq   Col %
    country_var Country     gender_var Gender
    Country     NZ          Gender     Female     44  54.321
                                       Male       37  45.679
                                       TOTAL      81   100.0
                South Korea Gender     Female     21  45.652
                                       Male       25  54.348
                                       TOTAL      46   100.0
                TOTAL       Gender     Female     99  48.529
                                       Male      105  51.471
                                       TOTAL     204   100.0
                USA         Gender     Female     34  44.156
                                       Male       43  55.844
                                       TOTAL      77   100.0

    Finally, round numbers
    """
    columns = row_vars + ['n', ]
    df_pre_pivot = pd.DataFrame(data, columns=columns)
    index_cols = []
    column_cols = []
    for var in row_vars:
        df_pre_pivot[get_pandas_friendly_name(var, '_var')] = var  ## e.g. country_var = Country
        cols2add = [get_pandas_friendly_name(var, '_var'), var]  ## e.g. [country_var, Country]
        index_cols.extend(cols2add)
    ## only add what is needed to fill gaps
    for i in range(n_row_fillers):
        df_pre_pivot[f'row_filler_var_{i}'] = BLANK
        df_pre_pivot[f'row_filler_{i}'] = BLANK
        index_cols.extend([f'row_filler_var_{i}', f'row_filler_{i}'])
    df_pre_pivot['metric'] = 'Freq'
    df_pre_pivot['n'] = df_pre_pivot['n'].astype(pd.Int64Dtype())
    if debug: print(df_pre_pivot)
    df_pre_pivots = [df_pre_pivot, ]
    column_cols = ['metric', ]  ## simple cf a cross_tab
    df = df_pre_pivot.pivot(index=index_cols, columns=column_cols, values='n')  ## missing rows e.g. if we have no rows for females < 20 in the USA, now appear as NAs so we need to fill them in df
    with no_silent_downcasting():
        df = infer_objects_no_copy(df.fillna(0)) ## needed so we can round values (can't round a NA). Also need to do later because of gaps appearing when pivoted then too
    if inc_col_pct:
        df_pre_pivot_inc_row_pct = get_df_pre_pivot_with_pcts(
            df, is_cross_tab=False, pct_type=PctType.COL_PCT, decimal_points=decimal_points, debug=debug)
        df_pre_pivots.append(df_pre_pivot_inc_row_pct)
    df_pre_pivot = pd.concat(df_pre_pivots)
    df_pre_pivot['__throwaway__'] = 'Metric'
    df = df_pre_pivot.pivot(index=index_cols, columns=['__throwaway__', ] + column_cols, values='n')
    with no_silent_downcasting():
        df = infer_objects_no_copy(df.fillna(0))
    df = df.astype(str)
    ## have to ensure all significant digits are showing e.g. 3.33 and 1.0 or 0.0 won't align nicely
    correct_string_dps = partial(correct_str_dps, decimal_points=decimal_points)
    df = df.map(correct_string_dps)
    return df


@dataclass(frozen=False, kw_only=True)
class FrequencyTableDesign(CommonDesign):
    """
    Args:
        row_variable_designs: list of Rows
        include_column_percent: if `True` add a column percentage column
    """
    row_variable_designs: list[Row] = DEFAULT_SUPPLIED_BUT_MANDATORY_ANYWAY

    include_column_percent: bool = False
    debug: bool = False
    verbose: bool = False

    @property
    def totalled_vars(self) -> list[str]:
        tot_vars = []
        for row_spec in self.row_variable_designs:
            tot_vars.extend(row_spec.self_and_descendant_totalled_vars)
        return tot_vars

    @property
    def max_row_depth(self) -> int:
        max_depth = 0
        for row_spec in self.row_variable_designs:
            row_depth = len(row_spec.self_and_descendant_vars)
            if row_depth > max_depth:
                max_depth = row_depth
        return max_depth

    def __post_init__(self):
        CommonDesign.__post_init__(self)
        row_vars = [spec.variable_name for spec in self.row_variable_designs]
        row_dupes = set()
        seen = set()
        for row_var in row_vars:
            if row_var in seen:
                row_dupes.add(row_var)
            else:
                seen.add(row_var)
        if row_dupes:
            raise ValueError(f"Duplicate top-level variable(s) detected in row dimension - {sorted(row_dupes)}")

    def get_row_df(self, cur, *, row_idx: int, decimal_points: int = 2) -> pd.DataFrame:
        """
        See cross_tab docs
        """
        row_spec = self.row_variable_designs[row_idx]
        totalled_variables = row_spec.self_and_descendant_totalled_vars
        row_vars = row_spec.self_and_descendant_vars
        data = get_data_from_spec(cur, dbe_spec=self.dbe_spec,
            source_table_name=self.source_table_name, table_filter_sql=self.table_filter_sql,
            all_variables=row_vars, totalled_variables=totalled_variables, debug=self.debug)
        n_row_fillers = self.max_row_depth - len(row_vars)
        df = get_all_metrics_df_from_vars(data, row_vars=row_vars, n_row_fillers=n_row_fillers,
            inc_col_pct=self.include_column_percent,
            decimal_points=decimal_points, debug=self.debug)
        return df

    def get_tbl_df(self, cur) -> pd.DataFrame:
        """
        See cross_tab docs
        """
        dfs = [self.get_row_df(cur, row_idx=row_idx, decimal_points=self.decimal_points)
            for row_idx in range(len(self.row_variable_designs))]
        df_t = dfs[0].T
        dfs_remaining = dfs[1:]
        for df_next in dfs_remaining:
            df_t = df_t.join(df_next.T, how='outer')
        df = df_t.T  ## re-transpose back so cols are cols and rows are rows again
        if self.debug: print(f"\nCOMBINED:\n{df}")
        ## Sorting indexes
        raw_df = get_raw_df(cur, dbe_spec=self.dbe_spec, source_table_name=self.source_table_name)
        order_rules_for_multi_index_branches = get_order_rules_for_multi_index_branches(self.row_variable_designs)
        ## ROWS
        unsorted_row_multi_index_list = list(df.index)
        sorted_row_multi_index_list = get_sorted_multi_index_list(
            unsorted_row_multi_index_list, order_rules_for_multi_index_branches=order_rules_for_multi_index_branches,
            sort_orders=self.sort_orders, raw_df=raw_df, has_metrics=False, debug=self.debug)
        sorted_row_multi_index = pd.MultiIndex.from_tuples(
            sorted_row_multi_index_list)  ## https://pandas.pydata.org/docs/user_guide/advanced.html
        sorted_col_multi_index_list = sorted(
            df.columns, key=lambda metric_label_and_metric: get_metric2order(metric_label_and_metric[1]))
        sorted_col_multi_index = pd.MultiIndex.from_tuples(sorted_col_multi_index_list)
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
        if self.debug:
            print(pd_styler.uuid)  ## A unique identifier to avoid CSS collisions; generated automatically.
            print(html)
        return HTMLItemSpec(
            html_item_str=html,
            output_item_type=OutputItemType.MAIN_TABLE,
            output_title=self.output_title,
            design_name=self.__class__.__name__,
            style_name=self.style_name,
        )
