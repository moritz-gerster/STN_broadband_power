"""Apply some functions on df."""
from os.path import basename, join

import numpy as np
import pandas as pd

import scripts.config as cfg


def export_localization_table(bands=cfg.BAND_LOCALIZATION, only_maxima=True,
                              keep_zero=False, add_normalized=True,
                              add_absolute=True):
    """
    Export df for localization plotting.
    bands: list
        List of bands to include in the table.
    only_maxima: bool
        Whether to create heatmap from all channels or maximum per hemi only.
        -> Only Maxima gives much more reasonable results and simpler stats.
    keep_zero: bool
        Whether to keep periodic zero power values when FOOOF was not fitted
        or replace with np.nan. Setting nan gives more resonable results.
    add_normalized: bool
        Gives much better results than total but worse than fooof.
    add_absolute: bool
        Seems to not work due to not-normalized log scale across patients.
        """
    df = pd.read_pickle(join(cfg.DF_PATH, cfg.DF_FOOOF))

    # filter df
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4', 'LFP_4-5', 'LFP_5-6',
               'LFP_6-7', 'LFP_7-8']
    df = df[(df.ch_reference == 'bipolar') & ~df.ch_bad & df.ch.isin(bip_chs)
            & df.mni_x.notna() & (df.cond.isin(['on', 'off']))
            & ~df.project.isin(['all'])]
    norm_mask = ((df.fm_params.isin([False]))
                 & (df.psd_kind.isin(['normalized'])))
    tot_mask = ((df.fm_params.isin(['broad']))
                & (df.psd_kind.isin(['standard'])))
    df_tot = df[tot_mask]
    df_norm = df[norm_mask]

    info_cols = ['subject', 'project', 'ch_nme', 'ch_hemisphere', 'cond',
                 'mni_x', 'mni_y', 'mni_z']
    keep_cols = info_cols.copy()
    total_pwr = '_abs_max_log'
    per_pwr = '_fm_powers_max_log'
    pwr_cols = []
    fm_cols = [f'{band}{per_pwr}' for band in bands]
    if add_absolute:
        abs_cols = [f'{band}{total_pwr}' for band in bands]
        keep_cols += abs_cols
        pwr_cols += abs_cols
    keep_cols += fm_cols
    pwr_cols += fm_cols
    keep_cols = list(set(keep_cols).intersection(set(df_tot.columns)))
    pwr_cols = list(set(pwr_cols).intersection(set(df_tot.columns)))
    _select_maxima(df_tot, pwr_cols, only_maxima=only_maxima,
                   keep_zero=keep_zero)
    df_tot = df_tot[keep_cols]

    if add_normalized:
        keep_cols = info_cols.copy()
        abs_cols = [f'{band}{total_pwr}' for band in bands]
        keep_cols += abs_cols
        keep_cols = list(set(keep_cols).intersection(set(df_norm.columns)))
        _select_maxima(df_norm, abs_cols, only_maxima=only_maxima,
                       keep_zero=keep_zero)
        df_norm = df_norm[keep_cols]

        # rename columns
        rename = {f'{band}{total_pwr}': f'{band}_normalized{total_pwr}'
                  for band in bands}
        pwr_cols += list(rename.values())
        df_norm.rename(columns=rename, inplace=True)
        df = pd.merge(df_tot, df_norm, on=info_cols)
    else:
        df = df_tot

    # if log scale causes negative values, shift all values to positive range
    # to prohibit negative values while keeping a normal distribution
    for pwr_col in pwr_cols:
        df[pwr_col] = df[pwr_col] + np.abs(min(df[pwr_col].min(), 0))

    # Change to wide format by adding condition in column name
    index = ['subject', 'ch_nme']
    df_pivot = df.pivot(index=index, columns='cond', values=pwr_cols)
    df_pivot = df_pivot.reset_index()
    new_cols = []
    for level, col in df_pivot.columns:
        if col == '':
            new_cols.append(level)
        else:
            new_cols.append(f"{level}_{col}")
    df_pivot.columns = new_cols
    df = pd.merge(df[info_cols], df_pivot, on=index)

    df = df[info_cols + [col for col in df.columns if col not in info_cols]]
    df.drop(columns='cond', inplace=True)
    df.drop_duplicates(subset=['subject', 'ch_nme'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # save as excel sheet
    df.to_excel(join(cfg.DF_PATH, 'localization_powers.xlsx'),
                index=False)


def _select_maxima(df, pwr_cols, only_maxima=True, keep_zero=False):
    if not only_maxima:
        return
    ch_max_cols = [f'ch_chmax_{pwr_col}' for pwr_col in pwr_cols]
    for ch_max_col in ch_max_cols:
        pwr_col = ch_max_col.replace('ch_chmax_', '')
        df.loc[~df[ch_max_col], pwr_col] = np.nan
        if not keep_zero:
            # also set 0 power to nan
            df.loc[df[pwr_col] == 0, pwr_col] = np.nan


if __name__ == "__main__":
    export_localization_table()
    print(f'{basename(__file__)} done.')