---
title: 'sofastats_lib: A Python package for basic statistical tests, report tables, and charting'
tags:
  - Python
  - statistics
  - charts
authors:
  - name: Grant R. Paton-Simpson
date: 22 March 2026
---

# Summary

`sofastats_lib` is a Python library for charting, report tables, and inferential statistical tests.
It is based on the design of the SOFA Statistics desktop application launched in 2009[^1].

`sofatstats_lib` provides a standard interface for connecting to CSV files and databases and generating report-ready self-contained HTML output.
In the case of statistical tests, result are also generated in an object (data class) readily consumed by Python programs.
Where it makes sense, there is also the option of worked example explanations of the results using the actual data used -
for example, of the Mann-Whitney U results.
Output can be themed by pre-existing styles
or by custom YAML-defined styles ([Making Custom Styles](https://sofastats.github.io/sofastats_lib/styles/)).

![Output Examples](https://sofastats.github.io/sofastats_lib/images/carousel.png){ width=95% }


# Statement of Need

Python is the most popular language for data science and analytics.
Python developers are well served with basic statistical tests and visualization options.
But there is a gap for an integrated set of tools for charting, report tables, and inferential statistical tests.
`sofastats_lib` aims to make such tools accessible to beginners
as well as expert analysts through:

1) a standardized API (see [API Documentation](https://sofastats.github.io/sofastats_lib/API))
2) detailed documentation ([README](https://sofastats.github.io/sofastats_lib/)
and [Data Preparation](https://sofastats.github.io/sofastats_lib/data_prep/)),
3) and a design informed by UX research
(see ["How UX Can Improve Your Python Project" by Grant and Charlotte Paton-Simpson](https://www.youtube.com/watch?v=5DDZa46g3Yc)).

[^1]: SOFA Statistics won the 2012 People's Choice Award in the New Zealand Open Source Awards
and was a finalist for the Best Open Source Project Award.


# State of the Field

Python is well-served with a very wide range of statistical functions
ranging from the very basic functions provided by the built-in [statistics](https://docs.python.org/3/library/statistics.html) library,
which supplies low-level functions like `mean`, `median`, `stdev`, and `linear_regression`,
through to the extensive and often advanced offerings of
`scipy.stats` ([Statistical functions (scipy.stats)](https://docs.scipy.org/doc/scipy/reference/stats.html))
and `statsmodels` ([statsmodels User Guide](https://www.statsmodels.org/stable/user-guide.html)).
Rounding out the options are PyMC which is for Bayesian statistics (not currently available in `sofastats_lib`)
and `pingouin` ([JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.01026);
[Pingouin Documentation](https://pingouin-stats.org/index.html)) which is most similar in purpose to `sofastats_lib`.

The rationale for `sofastats_lib` is to provide a unified experience when creating charts, report tables, and
inferential statistical test output. See the Examples section below.
As with `pingouin` raw statistical output is in the form of dataclasses
but `sofastats_lib` includes integrated charts, tables, and, where possible, workings as part of test output.
All output is available as standalone HTML which has advantages for more interactive charts
and for distribution of high-quality output. A final reason for making `sofastats_lib`
is to provide the underpinnings of a web GUI desktop application to replace SOFA Statistics.
Although `pingouin` is too different from the vision of `sofastats_lib`
to justify a "contribute" strategy instead of a "build" strategy there may be cross-pollination in the future.


# Software Design

## Python Library

The main design decision was to supply the code as a Python library defined by a pyproject.toml file
as is current best practice.
UV was used for dependency management and the CI process includes checks that the dependencies
can all be reconciled successfully on all versions of Python explicitly supported.
The goal is for people to be able to integrate `sofastats_lib` into existing Python projects,
Python being the main language used by Data Scientists and analysts generally. 

## Standardisation

To make `sofastats_lib` easier to consume and document,
it was decided to standardise the interface across charting, report tables, and statistical tests.
The CommonDesign dataclass was used as the base for all output design dataclasses
with the contract that they would all implement the `to_html_design()` method
which returns an `HTMLItemSpec` dataclass which in turn has `to_standalone_html()` and `to_result()` methods.
Having a common approach makes it easier to implement new charts, report tables, or statistical tests in the future.

## Clear Naming

Although contractions and abbreviations have been used in variable names
throughout the `sofastats_lib` codebase for brevity,
an effort was made to use the clearest possible names for variables and attributes in the surface API,
even if that lengthens names.

Examples:

* naming a parameter for filtering underlying source data "table_filter_sql" instead of "filt_str"
to make it more obvious that an appropriate SQL syntax must be followed
* using "database_engine_name" instead of "dbe_name"
* calling the dataclasses used for defining output "designs" and supplying all-in-one methods like "make_output".

## Parameter Grouping

When it came to parameter order there was a decision to make -
should parameters be ordered so those with default values came last, which would be the easiest approach to implement?
Or should parameters be ordered into logical groupings - for example, putting all data-related configuration together,
which would require a different approach to default values?
The latter was chosen to improve the user experience
and additional logic was added to enforce the supply of values lacking sensible defaults.

## Dataclass Interfaces

Dataclasses have been used throughout `sofastats_lib` as the main configuration inputs and returned values.
This approach provides clear interfaces and makes type hinting more informative.
It is also possible to add validation to inputs and to make the code more self-documenting than when,
for example, dictionaries are used.

## Organising Into Folders

Code has been organised into folders and subfolders as needed but there was an attempt to avoid excessive nesting.
For example, the output code for making bar charts is found in `sofastats_lib/output/charts/bar.py`.
The code for running a Mann-Whitney U test is in `sofastats_lib/output/stats/mann_whitney_u.py`.

## Making Custom Styles Easy to Configure

The styling system for `sofastats_lib` was designed so no Python was required, only YAML configuration.
The standard, built-in styles provide a model for how to make custom styles, and include the use of named items for colors.


# Research Impact Statement

`sofastats_lib` is based on the established SOFA Statistics desktop application
so although the library is too new to have an established record, the application it was based on
has been in use since 2009. SOFA Statistics has been downloaded over 350,000 times from SourceForge.
Additionally, it has been cited by hundreds[^1] of articles according to Google Scholar, including articles on
everything from bat virus phosphoproteins to concentrations of artificial sweeteners in wastewater.
SOFA Statistics has been widely used for teaching applied statistics to students
and video and text documentation has been produced to assist student labs - for example, a 135 page manual
[LAB MANUAL SOFA: Statistics Open For All - June 2017 – Edition 2.0 135 pages; George Self](https://ssric.calstate.edu/sites/default/files/2019-10/G_SELF_LabManual.pdf))

[^1]: In addition to the 88 articles which displayed the recommended standard citation
there were numerous examples that did not.
The exact number is difficult to extract because of the presence of an unrelated metric called SOFA,
but it is less than 480.

# AI Usage Disclosure

No AI tools were used.


# Features

[Output Gallery](https://sofastats.github.io/sofastats_lib/output_gallery.html)

## Charts

* Area Charts
* Multiple-Chart Area Charts
* Bar Charts
* Multiple-Chart Bar Charts
* Clustered Bar Charts
* Multiple-Chart Clustered Bar Charts
* Box Plots
* Clustered Box Plots
* Histograms
* Multiple-Chart Histograms
* Line Charts
* Multiple-Chart Line Charts
* Pie Charts
* Multiple-Chart Pie Charts
* Scatter Plots
* Multiple-Chart Scatter Plots
* By Series Scatter Plots
* Multiple-Chart By Series Scatter Plots

## Report Tables

* Frequency Tables
* Cross Tabs

## Inferential Statistical Tests

* ANOVA (One Way)
* Chi Square (including table of expected and observed values)
* Independent Samples T-Test
* Kruskal-Wallis H
* Mann-Whitney U
* Normality Test (for both independent and paired data)
* Paired Samples T-Test
* Pearson's R Correlation
* Spearman's R Correlation
* Wilcoxon Signed Ranks Test

# Examples

Full examples, including all required imports and data sources,
can be found at [sofastats_examples/scripts](https://github.com/sofastats/sofastats_lib/tree/main/src/sofastats_examples/scripts)

Below are examples of the actual configuration required to make output.
The goal of interface design was to minimise boilerplate and standardise across charting, report tables, and statistical tests.

## Chart

```python
from sofastats.output.charts.box_plot import ClusteredBoxplotChartDesign
...

design = ClusteredBoxplotChartDesign(
    csv_file_path=csv_file_path,
    output_file_path=output_folder / 'demo_multi_series_box_plot.html',
    output_title="Multi-Series Boxplot (black_pastel design)",
    show_in_web_browser=True,
    sort_orders_yaml_file_path=sort_orders_yaml_file_path,
    style_name='black_pastel',
    field_name='Age',
    category_field_name='Home Location Type',
    series_field_name='Country',
    series_sort_order=SortOrder.CUSTOM,
    category_sort_order=SortOrder.CUSTOM,
    box_plot_type=BoxplotType.INSIDE_1_POINT_5_TIMES_IQR,
    show_n_records=True,
    x_axis_font_size=12,
    decimal_points=3,
)
design.make_output()
```

![Clustered Box Plot Example](https://sofastats.github.io/sofastats_lib/images/clustered_box_plot_black_pastel_style.png){ width=75% }

## Report Table

```python
from sofastats.output.tables.cross_tab import CrossTabDesign
...

row_variables_design_1 = (
    Row(variable_name='Country',
        has_total=True, sort_order=SortOrder.CUSTOM,
        child=Row(
            variable_name='Home Location Type',
            has_total=True, sort_order=SortOrder.VALUE))
row_variables_design_2 = (
    Row(variable_name='Home Location Type',
        has_total=True, sort_order=SortOrder.CUSTOM))
row_variables_design_3 = Row(variable_name='Car')

col_variables_design_1 = (
    Column(variable_name='Sleep Group',
        has_total=True, sort_order=SortOrder.CUSTOM))
col_variables_design_2 = (
    Column(variable_name='Age Group',
        has_total=True, sort_order=SortOrder.CUSTOM,
        child=Column(
            variable_name='Handedness',
            has_total=True, sort_order=SortOrder.CUSTOM,
            pct_metrics=[Metric.ROW_PCT, Metric.COL_PCT]))
col_variables_design_3 = (
    Column(variable_name='Tertiary Qualifications',
        has_total=True, sort_order=SortOrder.CUSTOM))

design = CrossTabDesign(
    cur=sqlite_cur,
    database_engine_name=DbeName.SQLITE,  ## or just the string 'sqlite'
    source_table_name='people',
    table_filter_sql="WHERE Car IN ('Porsche', 'Toyota', 'Aston Martin')",
    output_file_path=output_folder / 'demo_main_cross_tab.html',
    output_title='Cross Tab from SQLite (Filtered by Car)',
    show_in_web_browser=True,
    sort_orders_yaml_file_path=sort_orders_yaml_file_path,
    row_variable_designs=[
        row_variables_design_1, row_variables_design_2, row_variables_design_3],
    column_variable_designs=[
        col_variables_design_1, col_variables_design_2, col_variables_design_3],
    style_name='default',
    decimal_points=2,
)
design.make_output()
```

## Statistical Test

### Close-Up View

![Cross Tab Example - Close-Up](https://sofastats.github.io/sofastats_lib/images/cross_tab_red_spirals_style_truncated.png){ width=75% }

### Complete View

Note - it may not make sense to make massive tables but the point is that `sofastats_lib` can generate whatever is configured.

![Cross Tab Example](https://sofastats.github.io/sofastats_lib/images/cross_tab_red_spirals_style.png){ width=75% }

```python
from sofastats.output.stats.anova import AnovaDesign
...

design = AnovaDesign(
    csv_file_path=csv_file_path,
    output_file_path=output_folder / 'demo_anova_age_by_country.html',
    output_title='ANOVA - Black Pastel Style',
    show_in_web_browser=True,
    style_name='black_pastel',
    grouping_field_name='Country',
    grouping_field_values=['South Korea', 'NZ', 'USA'],
    measure_field_name='Age',
    high_precision_required=False,
    decimal_points=3,
)
design.make_output()
```

![One-Way ANOVA Example](https://sofastats.github.io/sofastats_lib/images/anova_black_pastel_style_as_ordered.png){ width=75% }
