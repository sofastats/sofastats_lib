"""
Super overview *********************************************************************************************************

Multi-indexes can be flattened into lists of tuples, and those tuples can be sorted, converted back into multi-indexes,
and the df reindexed by the new, sorted multi-index, thus sorting the df as desired.

Overview ***************************************************************************************************************

E.g.
                Pet
      Cat                  Dog
      Age                  Age
 Young    Old         Young   Old

=>
[
    (Pet, Cat, Age, Young),
    (Pet, Cat, Age, Old),
    (Pet, Dog, Age, Young),
    (Pet, Dog, Age, Old),
]

The list above can easily be made like this: list(df.index) or list(df.columns) - too easy!!!

We can then associate each multi-index tuple with a derived sorting tuple which we use to do the actual sorting.
We make this sorting tuple in two steps.

1) We make an intermediate sorting tuple which includes the sort index of the variables
and the sort approach for the values e.g. by freq ascending
order_rules_for_multi_index_branches

2) We work out the sorted position of the values using the sort approach + either the value
or the frequency associated with the value

Then we sort the original list of tuples using the associated sorting tuple as the sort key. QED!
We have our sorted list of tuples ready to convert back into a multi-index, and reapply as the df index.


E.g. imagine we start with an index like this:
[
  ['a', 'cat', 'banana', 123],
  ['b', 'dog', 'apple', 989],
]

We then apply a function to the list to create a sorting-ready tuple. E.g.:

['a', 'cat', 'banana', 123]
=>
(1, 12, 2, 5)  ## Note - not necessarily alphabetical -
might be based on the frequencies associated with the values or something
and
['b', 'dog', 'apple', 989]
=>
(0, 3, 6, 12)  ## also, just for the purposes of illustration, imagine if this were the result of the same function

Simple sorting would put (0, 3, 6, 12) before (1, 12, 2, 5) and so we would end up with the following sorted multi-index:
[
  ['b', 'dog', 'apple', 989],
  ['a', 'cat', 'banana', 123],
]

Details ****************************************************************************************************************

So how should sorting occur?

There are two separate sorts of thing being sorted. Variables and values e.g. 'Age Group' and '< 20', '20-29' etc.
Variables get their sort order from the order in which they are configured in the table design.
For example, if we have the following variable design:

     Age Group                   Web Browser
         |                           |
         |              ----------------------------
         |              |                          |
         |          Age Group                      Car

we end up with a df like:

     Age Group                                       Web Browser
Young  Middle  Old                       Firefox                                      Chrome
                           Age Group               Car                   Age Group               Car
                      Young  Middle  Old    Tesla  Mini  Porsche    Young  Middle  Old    Tesla  Mini  Porsche
Freq   Freq    Freq   Freq   Freq   Freq    Freq   Freq  Freq       Freq   Freq   Freq    Freq   Freq  Freq
--------------------------------------------------------------------------------------------------------------

and a multi-index like:
[
  ...
  ('Age Group', 'Middle', '__blank__', '__blank__', 'Freq'),
  ...
  ('Web Browser', 'Chrome', 'Car', 'Porsche', 'Freq'),
  ...
]

We need to first configure the sort order:

Remember, there are two parts - variables, and their values.
The variables are simple index values based on the order they were configured.
E.g. at the top level, Age Group is 0 and Web Browser 1.
Under Web Browser, Age Group is 0, and Car is 1.
For values, we have two sort order options:
 * by value e.g. 'Apple', then 'Banana' etc.
 * or by frequency (subdivided into either increasing or decreasing).
We also have the metric where Freq then Row % then Col %.

So ... working down the chains, our task is to generate a key and a value where the key is the chain,
and the value is the sorting tuple (order_rules_for_multi_index_branches).
The sorting tuple has (variable sortable, value sortable, ...

The three chains: Age Group, Web Browser > Age Group, Web Browser > Car

Step 1 - chains to chain-tuple as key and order_rules_for_multi_index_branches tuple as value:

Chain 1 (just Age Group, we can ignore blanks and the metrics)
                  Age Group  Age Group  (only need to define sort order for the values, not the metrics
                   variable    value                             - they have a standardised order)
                      |         |
{                     v         v
    ('Age Group', ): (0, Sort.VALUE),

Chain 2 (Web Browser > Age Group)
                                    Web      Web     Age      Age
                                  Browser  Browser  Group    Group
                                 variable   value  variable  value
                                     |        |       |       |
                                     v        v       v       v
    ('Web Browser', 'Age Group', ): (1,  Sort.VALUE,  0, Sort.VALUE),

Chain 3 (Web Browser > Car)
                                 Web      Web
                               Browser  Browser    Car      Car
                              variable   value   variable  value
                                  |        |        |    |
                                  v        v        v    v
    ('Web Browser', 'Car', ):    (1, Sort.LABEL,    1, Sort.LABEL),
}

Then apply the sort order knowing it is variable, value, ... (skipping __blank__ - leave it as is), measure:

('Age Group', 'Middle', '__blank__', '__blank__', 'Freq'),

1) From the index row tuple, get the branch of _variable_ keys e.g.
('Web Browser', 'Firefox', 'Car', 'AUDI', Metric.FREQ)
=>
('Web Browser', 'Car', )

2) Find the matching sort order for that branch of variables key
(in this case, a branch from Web Browser to Car).

3) Apply that sort order to the original index row.

Example Usage **********************************************************************************************************

order_rules_for_row_multi_index_branches = get_order_rules_for_multi_index_branches(self.row_variable_designs)
order_rules_for_col_multi_index_branches = get_order_rules_for_multi_index_branches(self.column_variable_designs)
## COLS
unsorted_col_multi_index_list = list(df.columns)
sorted_col_multi_index_list = get_sorted_multi_index_list(
    unsorted_col_multi_index_list, order_rules_for_multi_index_branches=order_rules_for_col_multi_index_branches,
    var_labels=self.data_labels, raw_df=raw_df, has_metrics=True, debug=self.debug)
sorted_col_multi_index = pd.MultiIndex.from_tuples(sorted_col_multi_index_list)
## ROWS
unsorted_row_multi_index_list = list(df.index)
sorted_row_multi_index_list = get_sorted_multi_index_list(
    unsorted_row_multi_index_list, order_rules_for_multi_index_branches=order_rules_for_row_multi_index_branches,
    var_labels=self.data_labels, raw_df=raw_df, has_metrics=False, debug=self.debug)
sorted_row_multi_index = pd.MultiIndex.from_tuples(sorted_row_multi_index_list)
df = df.reindex(index=sorted_row_multi_index, columns=sorted_col_multi_index)
"""
from functools import partial
from itertools import count

import pandas as pd

from sofastats.conf.main import SortOrder, SortOrderSpecs
from sofastats.output.tables.interfaces import BLANK, TOTAL, Column, Metric, Row

pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 750)

def index_row2branch_of_vars_key(orig_tuple: tuple, *, debug=False) -> tuple:
    """
    We need this so we can look up the sorting required for this variable sequence.

               |                      |
               v                      v
    e.g. ('Web Browser', 'Firefox', 'Car', 'AUDI', Metric.FREQ)
    =>
    ('Web Browser', 'Car', )

and
          |
          v
    ('Age Group', '20-29', BLANK, BLANK, Metric.FREQ)
    =>
    ('Age Group', )
    """
    items = []
    for idx in count(0, 2):
        try:
            item = orig_tuple[idx]
        except IndexError:
            break
        if item in [member.value for member in Metric] or item == BLANK:
            break
        items.append(item)
        if debug:
            print(f"{idx=}")
        if idx > 1_000:
            raise Exception(
                "Seriously?! Over 1,000 items in multi-index tuple?! Something has gone badly wrong! Abort!!")
    items = tuple(items)
    if debug:
        print(f"{orig_tuple} => {items}")
    return tuple(items)

def _by_freq(variable: str, label: str, df: pd.DataFrame, filters: tuple[tuple[str, str]] | None = None, *,
        increasing=True) -> tuple[int, float]:
    """
    Args:
        filts: [('browser', 'Firefox'), ...] or [('agegroup', '< 20'), ...]
    """
    if label == TOTAL:
        sort_val = (1, 'anything ;-)')
    else:
        df_filt = df.copy().loc[df[variable] == label]
        for filt_variable, filt_val_label in filters:
            df_filt = df_filt.loc[df_filt[filt_variable] == filt_val_label]
        freq = len(df_filt)
        if increasing:
            sort_val = freq
        else:
            try:
                sort_val = 1 / freq
            except ZeroDivisionError as e:
                sort_val = 1.1  ## so always at end after labels with freq of at least one (given we are decreasing)
        sort_val = (0, sort_val)
    return sort_val

def _get_branch_of_variables_key(index_with_labels: tuple, *, debug=False) -> tuple:
    """
    How should we sort the multi-index? The details are configured against variable branch trees. E.g.
    {
        ('age', ): (0, Sort.VAL),
        ('browser', 'age', ): (1, Sort.LBL, 0, Sort.VAL),
        ('browser', 'car', ): (1, Sort.LBL, 1, Sort.LBL),
    }
    So we need to extract the key (in the form of a branch of variables) from the multi-index
    so we can read the sort configuration for it. E.g.
    ('browser', 'Firefox', 'car', 'AUDI', Metric.FREQ)
    =>
    ('browser', 'car', )
    """
    index_with_vars = []
    for i, item in enumerate(index_with_labels):
        vars_finished = ((item == BLANK) or i == len(index_with_labels) - 1)
        if vars_finished:
            index_with_vars.append(item)
            continue
        index_with_vars.append(item)
    branch_of_variables_key = index_row2branch_of_vars_key(tuple(index_with_vars), debug=debug)
    return branch_of_variables_key

def get_metric2order(metric: Metric) -> int:
    return {Metric.FREQ: 1, Metric.ROW_PCT: 2, Metric.COL_PCT: 3}[metric]

def get_order_rules_for_multi_index_branches(rows_or_columns: list[Row] | list[Column]) -> dict:
    """
    Note - to test Sort.INCREASING and Sort.DECREASING I'll need to manually check what expected results should be (groan)

    Note - because, below the top-level, only chains are allowed (not trees)
    the index for any variables after the first (top-level) are always 0.
    """
    orders = {}
    for top_level_idx, dim_type_spec in enumerate(rows_or_columns):
        dim_vars = tuple(dim_type_spec.self_and_descendant_vars)
        sort_details = []
        for chain_idx, dim_spec in enumerate(dim_type_spec.self_and_descendants):
            idx2use = top_level_idx if chain_idx == 0 else 0
            sort_details.extend([idx2use, dim_spec.sort_order])
        orders[dim_vars] = tuple(sort_details)
    return orders

def get_tuple_for_sorting(orig_index_tuple: tuple, *, order_rules_for_multi_index_branches: dict,
        sort_orders: SortOrderSpecs, raw_df: pd.DataFrame, has_metrics: bool, debug=False) -> tuple:
    """
    Use this method for the key arg for sorting e.g. sorted(unsorted_multi_index_list, key=multi_index_sort_fn)

    E.g.
    ('Age Group', '<20', '__blank__', '__blank__', 'Freq')
    => (we get key to branch of variables key ('Age Group', ) and then lookup sorting e.g. (0, Sort.VALUE))
    (0, 1, 0, 0, 1) given 1 is the val for '< 20' and 0 is our index for Freq (cf Row % and Col %)

    ('Age Group', '65+', '__blank__', '__blank__', 'Row %')
    =>
    (0, 5, 0, 0, 1) given 5 is the val for '65+' and 1 is our index for Row %
    """
    max_idx = len(orig_index_tuple) - 1
    metric_idx = max_idx if has_metrics else None
    branch_of_variables_key = _get_branch_of_variables_key(index_with_labels=orig_index_tuple)
    order_rule = order_rules_for_multi_index_branches[branch_of_variables_key]  ## e.g. (1, Sort.LBL, 0, Sort.INCREASING)
    list_for_sorting = []
    variable_value_pairs = []  ## so we know what filters apply depending on how far across the index we have come e.g. if we have passed Gender Female then we need to filter to that
    if debug:
        print(f"{orig_index_tuple=}; {max_idx=}; {order_rule=}")
    for idx in count():
        if idx > max_idx:
            break
        ## get order for sorting
        is_metric_idx = (idx == metric_idx) if has_metrics else False
        is_var_idx = (idx % 2 == 0 and not is_metric_idx)
        is_val_idx = not (is_var_idx or is_metric_idx)
        if is_var_idx:
            variable_label = orig_index_tuple[idx]
            if variable_label == BLANK:
                variable_order = 0  ## never more than one BLANK below a parent so no sorting occurs - so 0 as good as anything else
            else:
                variable_order = order_rule[idx]
            if debug:
                print(f"{variable_label=}; {variable_order=}")
            list_for_sorting.append(variable_order)
        elif is_val_idx:
            """
            Because we want TOTAL to come last we namespace everything else with 0, and TOTAL with 1. Elegant 😙🤌 
            """
            value = orig_index_tuple[idx]
            if value == BLANK:
                value_order = (0, "doesn't matter - doesn't splay below this so nothing to sort beyond the order already set so far in the ordering tuple")  ## never more than one BLANK in vals (because all the rest to the right will also be BLANKs and there will be nothing to sort within the parent branch this BLANK was under) so not more than one to sort so sort order doesn't matter
            else:
                variable = orig_index_tuple[idx - 1]
                value_order_rule = order_rule[idx]
                if value_order_rule == SortOrder.VALUE:
                    if value != TOTAL:
                        value_order = (0, value)
                    else:  ## want TOTAL last
                        value_order = (1, 'anything - the 1 is enough to ensure sort order')
                elif value_order_rule == SortOrder.CUSTOM:
                    ## use supplied sort order
                    try:
                        values_in_order = sort_orders[variable]
                    except KeyError:
                        raise Exception(f"You wanted the values in variable '{variable}' to have a custom sort order "
                            "but I couldn't find a sort order from what you supplied. "
                            "Please fix the sort order details or use another approach to sorting.")
                    if value != TOTAL:
                        value2order = {val: order for order, val in enumerate(values_in_order)}
                        try:
                            idx_for_ordered_position = value2order[value]
                        except KeyError:
                            raise Exception(f"The custom sort order you supplied for values in variable '{variable}' "
                                f"didn't include value '{value}' so please fix that and try again.")
                        value_order = (idx_for_ordered_position, value)
                    else:  ## want TOTAL last
                        idx_for_ordered_position_of_total = len(values_in_order) - 1
                        value_order = (idx_for_ordered_position_of_total,
                            'anything - the idx_for_ordered_position_of_total is enough to ensure sort order')
                elif value_order_rule in (SortOrder.INCREASING, SortOrder.DECREASING):
                    increasing = (value_order_rule == SortOrder.INCREASING)
                    filters = tuple(variable_value_pairs)
                    value_order = _by_freq(variable, value, df=raw_df, filters=filters, increasing=increasing)  ## can't use df as arg for cached function  ## want TOTAL last
                else:
                    raise ValueError(f"Unexpected value order spec ({value_order_rule})")
                variable_value_pairs.append((variable, value))
            list_for_sorting.append(value_order)
        elif has_metrics and is_metric_idx:
            metric = orig_index_tuple[idx]
            metric_order = get_metric2order(metric)
            list_for_sorting.append(metric_order)
        else:
            raise ValueError(f"Unexpected item index ({idx=}) when getting tuple for sorting")
    tuple_for_sorting = tuple(list_for_sorting)
    return tuple_for_sorting

def get_sorted_multi_index_list(unsorted_multi_index_list: list[tuple], *, order_rules_for_multi_index_branches: dict,
        sort_orders: SortOrderSpecs, raw_df: pd.DataFrame, has_metrics: bool, debug=False) -> list[tuple]:
    """
    See module doc string for fuller explanation.

    ('Web Browser', 'Firefox', 'Car', 'AUDI', Metric.FREQ)

    1) Get from the index row tuple to the branch-of-variable keys e.g.
    ('Web Browser', 'Firefox', 'Car', 'AUDI', Metric.FREQ)
    =>
    ('Web Browser', 'Car', )

    2) Find the matching sort order rule for that branch of variables key
    (in this case, a branch from Web Browser to Car).

    3) Derive a sortable tuple from that original index row according to the sort order rule (see get_tuple_for_sorting)

    4) Sort by the sortable tuple

    Args:
        order_rules_for_multi_index_branches: e.g.
          {
            ('Age Group', ): (0, Sort.VALUE),
            ('Web Browser', 'Age Group', ): (1, Sort.VALUE, 0, Sort.VALUE),
            ('Web Browser', 'Car', ): (1, Sort.VALUE, 1, Sort.VALUE),
          }
        raw_df: e.g.
                  id  Age Group       Web Browser  Car
          0        1       < 20           Firefox    7
          ...    ...        ...               ...  ...
          1499  1500        65+  Internet Explorer   8
    """
    ## get sort function
    multi_index_sort_fn = partial(get_tuple_for_sorting,
        order_rules_for_multi_index_branches=order_rules_for_multi_index_branches,
        sort_orders=sort_orders, raw_df=raw_df, has_metrics=has_metrics, debug=debug)
    ## apply sort function
    sorted_multi_index_list = sorted(unsorted_multi_index_list, key=multi_index_sort_fn)
    if debug:
        for row in sorted_multi_index_list:
            print(row)
    return sorted_multi_index_list
