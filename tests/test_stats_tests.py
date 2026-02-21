"""
Strategy - for each statistical test, compare the result against two trusted sources:
1) the original stats.py
2) the widely used and scrutinised SOFA Statistics code
"""
from functools import partial
from pathlib import Path
import tempfile

import pandas as pd

from sofastats.conf.main import SortOrder
from sofastats.output.stats.anova import AnovaDesign
from sofastats.output.stats.chi_square import ChiSquareDesign
from sofastats.output.stats.independent_t_test import IndependentTTestDesign
from sofastats.output.stats.kruskal_wallis_h import KruskalWallisHDesign
from sofastats.output.stats.mann_whitney_u import MannWhitneyUDesign
from sofastats.output.stats.normality import NormalityDesign
from sofastats.output.stats.paired_t_test import PairedTTestDesign
from sofastats.output.stats.pearsons_r import PearsonsRDesign
from sofastats.output.stats.spearmans_r import SpearmansRDesign
from sofastats.output.stats.wilcoxon_signed_ranks import WilcoxonSignedRanksDesign
from tests.conf import people_csv_fpath, sort_orders_yaml_file_path
from tests.reference_stats_library import (
    akurtosis as stats_kurtosis,
    anormaltest as stats_normaltest,
    anova as stats_anova,
    askew as stats_askew,
    askewtest as stats_skewtest,
    chisquare_df_corrected as stats_chi_square,
    kurtosistest as stats_kurtosistest,
    lkruskalwallish as stats_kruskalwallish,
    lmannwhitneyu as stats_mannwhitneyu,
    lpearsonr as stats_pearsonr,
    lspearmanr as stats_spearmanr,
    lttest_ind as stats_ttest_ind,
    lttest_rel as stats_ttest_rel,
    lwilcoxont as stats_wilcoxont,
)

round_to_11dp = partial(round, ndigits=11)

def test_anova():
    design = AnovaDesign(
        csv_file_path=people_csv_fpath,
        grouping_field_name='Country',
        grouping_field_values=['South Korea', 'NZ', 'USA'],
        measure_field_name='Age',
        high_precision_required=False,
    )
    # design.make_output()
    result = design.to_result()
    print(result)
    df_raw = pd.read_csv(design.csv_file_path)
    df = df_raw.loc[
        df_raw[design.grouping_field_name].isin(design.grouping_field_values),
        [design.grouping_field_name, design.measure_field_name]]
    data_south_korea = df.loc[df[design.grouping_field_name] == 'South Korea', design.measure_field_name].tolist()
    data_nz = df.loc[df[design.grouping_field_name] == 'NZ', design.measure_field_name].tolist()
    data_usa = df.loc[df[design.grouping_field_name] == 'USA', design.measure_field_name].tolist()
    stats_f, stats_prob = stats_anova(data_south_korea, data_nz, data_usa)
    assert round_to_11dp(result.F) == round_to_11dp(stats_f)  ## 1.25871675527 88687 ~= 1.25871675527 92649
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)  ## 0.284123842228 97413 ~= 0.284123842228 84

def test_chi_square():
    orig_csv_file_path = people_csv_fpath
    df_unfiltered = pd.read_csv(orig_csv_file_path)
    df = df_unfiltered.loc[
        (df_unfiltered['Age Group'].isin(['<20', '20 to <30'])) & (df_unfiltered['Country'].isin(['Denmark', 'NZ']))]
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        df.to_csv(temp.name, index=False)
        csv_file_path = temp.name
        design = ChiSquareDesign(
            csv_file_path=csv_file_path,
            sort_orders_yaml_file_path=sort_orders_yaml_file_path,
            variable_a_name='Age Group', variable_a_sort_order=SortOrder.CUSTOM,
            variable_b_name='Country', variable_b_sort_order=SortOrder.CUSTOM,
        )
        # design.make_output()
        result = design.to_result()
    print(result)

    under_20_nz_freq = len(df.loc[(df['Age Group'] == '<20') & (df['Country'] == 'NZ')])
    under_20_denmark_freq = len(df.loc[(df['Age Group'] == '<20') & (df['Country'] == 'Denmark')])
    twenty_to_under_30_nz_freq = len(df.loc[(df['Age Group'] == '20 to <30') & (df['Country'] == 'NZ')])
    twenty_to_under_30_denmark_freq = len(df.loc[(df['Age Group'] == '20 to <30') & (df['Country'] == 'Denmark')])

    total_freq = len(df)

    nz_freq = len(df.loc[df['Country'] == 'NZ'])
    denmark_freq = len(df.loc[df['Country'] == 'Denmark'])
    nz_fraction = nz_freq / total_freq
    denmark_fraction = denmark_freq / total_freq

    under_20_freq = len(df.loc[df['Age Group'] == '<20'])
    twenty_to_under_30_freq = len(df.loc[df['Age Group'] == '20 to <30'])
    under_20_fraction = under_20_freq / total_freq
    twenty_to_under_30_fraction = twenty_to_under_30_freq / total_freq

    expected_under_20_nz_freq = total_freq * under_20_fraction * nz_fraction
    expected_under_20_denmark_freq = total_freq * under_20_fraction * denmark_fraction
    expected_twenty_to_under_30_nz_freq = total_freq * twenty_to_under_30_fraction * nz_fraction
    expected_twenty_to_under_30_denmark_freq = total_freq * twenty_to_under_30_fraction * denmark_fraction

    observed_freqs_by_country_within_age_group = [
        under_20_nz_freq, under_20_denmark_freq,
        twenty_to_under_30_nz_freq, twenty_to_under_30_denmark_freq]
    expected_freqs_by_country_within_age_group = [
        expected_under_20_nz_freq, expected_under_20_denmark_freq,
        expected_twenty_to_under_30_nz_freq, expected_twenty_to_under_30_denmark_freq]

    n_variable_a_vals = len(df['Age Group'].unique())
    n_variable_b_vals = len(df['Country'].unique())
    degrees_of_freedom = (n_variable_a_vals - 1) * (n_variable_b_vals - 1)
    stats_chisq, stats_prob = stats_chi_square(
        f_obs=observed_freqs_by_country_within_age_group, f_exp=expected_freqs_by_country_within_age_group,
        df=degrees_of_freedom)

    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)
    assert round_to_11dp(result.chi_square) == round_to_11dp(stats_chisq)

def test_independent_t_test():
    design = IndependentTTestDesign(
        csv_file_path=people_csv_fpath,
        grouping_field_name='Country',
        group_a_value='South Korea',
        group_b_value='USA',
        measure_field_name='Age',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df = df_raw.loc[
        df_raw[design.grouping_field_name].isin([design.group_a_value, design.group_b_value]),
        [design.grouping_field_name, design.measure_field_name]]
    data_south_korea = df.loc[df[design.grouping_field_name] == 'South Korea', design.measure_field_name].tolist()
    data_usa = df.loc[df[design.grouping_field_name] == 'USA', design.measure_field_name].tolist()

    t, stats_prob = stats_ttest_ind(a=data_south_korea, b=data_usa)

    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)
    assert round_to_11dp(result.t) == round_to_11dp(t)

def test_kruskal_wallis_h():
    design = KruskalWallisHDesign(
        csv_file_path=people_csv_fpath,
        grouping_field_name='Country',
        grouping_field_values=['South Korea', 'NZ', 'USA'],
        measure_field_name='Age',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df = df_raw.loc[
        df_raw[design.grouping_field_name].isin(design.grouping_field_values),
        [design.grouping_field_name, design.measure_field_name]]
    data_south_korea = df.loc[df[design.grouping_field_name] == 'South Korea', design.measure_field_name].tolist()
    data_nz = df.loc[df[design.grouping_field_name] == 'NZ', design.measure_field_name].tolist()
    data_usa = df.loc[df[design.grouping_field_name] == 'USA', design.measure_field_name].tolist()
    h, stats_prob = stats_kruskalwallish(data_south_korea, data_nz, data_usa)

    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)
    assert round_to_11dp(result.h) == round_to_11dp(h)

def test_mann_whitney_u():
    design = MannWhitneyUDesign(
        csv_file_path=people_csv_fpath,
        grouping_field_name='Country',
        group_a_value='South Korea',
        group_b_value='NZ',
        measure_field_name='Weight Time 1',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df = df_raw.loc[
        df_raw[design.grouping_field_name].isin([design.group_a_value, design.group_b_value]),
        [design.grouping_field_name, design.measure_field_name]]
    data_south_korea = df.loc[df[design.grouping_field_name] == 'South Korea', design.measure_field_name].tolist()
    data_nz = df.loc[df[design.grouping_field_name] == 'NZ', design.measure_field_name].tolist()

    small_u, stats_prob = stats_mannwhitneyu(x=data_south_korea, y=data_nz)

    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)
    assert round_to_11dp(result.small_u) == round_to_11dp(small_u)

def test_normality():
    design = NormalityDesign(
        csv_file_path=people_csv_fpath,
        style_name='default',
        variable_a_name='Age',
        variable_b_name='Weight Time 2',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df = df_raw.dropna(subset=[design.variable_a_name, design.variable_b_name])
    df['a_minus_b'] = df[design.variable_a_name] - df[design.variable_b_name]
    data_a_minus_b = df['a_minus_b'].tolist()

    k2, stats_prob_array = stats_normaltest(data_a_minus_b)
    z_skew, _unused = stats_skewtest(data_a_minus_b)
    c_skew = float(stats_askew(data_a_minus_b))
    z_kurtosis, unused = stats_kurtosistest(data_a_minus_b)
    c_kurtosis_without_fischer_adjustment = float(stats_kurtosis(data_a_minus_b))
    c_kurtosis = c_kurtosis_without_fischer_adjustment - 3.0

    assert round_to_11dp(result.k2) == round_to_11dp(k2)
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob_array[0])
    assert round_to_11dp(result.z_skew) == round_to_11dp(z_skew)
    assert round_to_11dp(result.c_skew) == round_to_11dp(c_skew)
    assert round_to_11dp(result.z_kurtosis) == round_to_11dp(float(z_kurtosis))
    assert round_to_11dp(result.c_kurtosis) == round_to_11dp(float(c_kurtosis))

def test_paired_t_test():
    design = PairedTTestDesign(
        csv_file_path=people_csv_fpath,
        variable_a_name='Weight Time 1',
        variable_b_name='Weight Time 2',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df_filtered = df_raw.dropna(subset=[design.variable_a_name, design.variable_b_name]).copy()
    df = df_filtered[[design.variable_a_name, design.variable_b_name]].copy()
    df.rename(columns={design.variable_a_name: 'var_a', design.variable_b_name: 'var_b'}, inplace=True)
    sample_a = []
    sample_b = []
    for row in df.itertuples():
        sample_a.append(row.var_a)
        sample_b.append(row.var_b)
    t, stats_prob = stats_ttest_rel(a=sample_a, b=sample_b)

    assert round_to_11dp(result.t) == round_to_11dp(t)
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)

def test_pearsons_r():
    design = PearsonsRDesign(
        csv_file_path=people_csv_fpath,
        variable_a_name='Age',
        variable_b_name='Weight Time 1',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df_filtered = df_raw.dropna(subset=[design.variable_a_name, design.variable_b_name]).copy()
    df = df_filtered[[design.variable_a_name, design.variable_b_name]].copy()
    df.rename(columns={design.variable_a_name: 'var_x', design.variable_b_name: 'var_y'}, inplace=True)
    sample_x = []
    sample_y = []
    for row in df.itertuples():
        sample_x.append(row.var_x)
        sample_y.append(row.var_y)
    r, stats_prob = stats_pearsonr(x=sample_x, y=sample_y)

    assert round_to_11dp(result.r) == round_to_11dp(r)
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)

def test_spearmans_r():
    design = SpearmansRDesign(
        csv_file_path=people_csv_fpath,
        variable_a_name='Age',
        variable_b_name='Weight Time 1',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df_filtered = df_raw.dropna(subset=[design.variable_a_name, design.variable_b_name]).copy()
    df = df_filtered[[design.variable_a_name, design.variable_b_name]].copy()
    df.rename(columns={design.variable_a_name: 'var_x', design.variable_b_name: 'var_y'}, inplace=True)
    sample_x = []
    sample_y = []
    for row in df.itertuples():
        sample_x.append(row.var_x)
        sample_y.append(row.var_y)
    r, stats_prob = stats_spearmanr(x=sample_x, y=sample_y)

    assert round_to_11dp(result.r) == round_to_11dp(r)
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)

def test_wilcoxon_signed_ranks():
    design = WilcoxonSignedRanksDesign(
        csv_file_path=people_csv_fpath,
        variable_a_name='Weight Time 1',
        variable_b_name='Weight Time 2',
    )
    # design.make_output()
    result = design.to_result()
    print(result)

    df_raw = pd.read_csv(design.csv_file_path)
    df_filtered = df_raw.dropna(subset=[design.variable_a_name, design.variable_b_name]).copy()
    df = df_filtered[[design.variable_a_name, design.variable_b_name]].copy()
    df.rename(columns={design.variable_a_name: 'var_x', design.variable_b_name: 'var_y'}, inplace=True)
    sample_x = []
    sample_y = []
    for row in df.itertuples():
        sample_x.append(row.var_x)
        sample_y.append(row.var_y)
    t, stats_prob = stats_wilcoxont(x=sample_x, y=sample_y)

    assert round_to_11dp(result.t) == round_to_11dp(t)
    assert round_to_11dp(result.p) == round_to_11dp(stats_prob)

if __name__ == '__main__':
    pass
    # test_anova()
    # test_chi_square()
    # test_independent_t_test()
    # test_kruskal_wallis_h()
    # test_mann_whitney_u()
    # test_normality()
    # test_paired_t_test()
    # test_pearsons_r()
    # test_spearmans_r()
    # test_wilcoxon_signed_ranks()
