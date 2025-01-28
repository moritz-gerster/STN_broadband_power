"""Apply functions to dataframe."""
import warnings
from os.path import basename, join

import numpy as np
import pandas as pd

import scripts.config as cfg
from specparam import SpectralModel
from scripts.utils import _ignore_warnings


CH_CHOICES = ['ch_choice_amp', 'ch_dist_sweet_inside', 'ch_adj_sweet_inside',
              'ch_dist_sweet', 'ch_dist_sweet_mean',
              'ch_adj_sweet', 'ch_adj_sweet_mean',
              'ch_dist_beta_high_max_off', 'ch_adj_beta_high_max_off']


def organize_df(bands=cfg.BANDS.keys()):
    """Apply multiple functions to power spectrum dataframe.

    Patient demographics and symptoms, band powers, channel selection, etc."""
    df = pd.read_pickle(join(cfg.DF_PATH, cfg.DF_FOOOF_RAW))
    _correct_types(df)
    _correct_sex(df)
    _get_neumann_days_afer_surgery(df)
    _same_amp_cond(df)
    _add_ch_colors(df)
    df = _updrs_add_bradyrigid(df)
    df = _updrs_ipsi_contra(df)
    _fill_missing_updrs(df)
    df = _symptom_dominant_side(df)
    _sweet_spot_distance(df)
    _sweetest_adjacent_chs(df)
    _exclude_bad_fits(df)
    _absolute_band_power(df, bands=bands)
    _subtract_noise_from_power(df)
    df = _fooof_band_power(df, bands=bands)  # slow
    if 'FTG' in bands:
        df = _add_FTGs(df)
    df = _area_under_curve_power(df, bands=bands)
    df = _periodic_integrated_power(df, bands=bands)  # slow
    _select_max_band(df, bands=cfg.BAND_LOCALIZATION)
    _channel_choice(df)
    _add_psd_means(df)
    df = _subtract_offon(df)
    df = _brady_severity(df)
    _add_psd_min_max(df)  # after offon added
    _set_bad_chs_nan(df)  # before averaging LFP_mean chs
    df = _add_ch_means(df)  # slow
    df = _organize_df(df)  # delete and sort columns
    df = _add_pooled_projects(df)
    _add_project_names(df)
    _add_new_subject_names(df)
    _correct_types(df)  # df needs to be corrected again
    save_path = join(cfg.DF_PATH, cfg.DF_FOOOF)
    df.to_pickle(save_path)
    print(f"{basename(__file__).strip('.py')} done.")


def _add_pooled_projects(df):
    df_all = df.copy()
    df_all['project'] = 'all'
    df = pd.concat([df, df_all], ignore_index=True)
    return df


def _add_project_names(df):
    """Add project names to df."""
    df['project_nme'] = df.project.map(cfg.PROJECT_DICT)
    df['color'] = df.project.map(cfg.COLOR_DIC)


def _add_new_subject_names(df):
    projects_short = {'Neumann': 'Ber', 'Tan': 'Oxf', 'Litvak': 'Lon',
                      'Florin': 'Du1', 'Hirschmann': 'Du2'}
    sub_dict = {}
    for project in cfg.RECORDINGS:
        old_names = df[(df.project == project)
                       & (df.bids_task != 'noise')].subject.unique()
        new_names = [f'{projects_short[project]}-{i:02d}'
                     for i in range(1, len(old_names) + 1)]
        proj_dict = dict(zip(old_names, new_names))
        sub_dict.update(proj_dict)
        # rename emptyroom
        old_names = df[(df.project == project)
                       & (df.bids_task == 'noise')].subject.unique()
        new_names = [nme.replace(project[:3], projects_short[project])
                     for nme in old_names]
        proj_dict = dict(zip(old_names, new_names))
        sub_dict.update(proj_dict)
    df['subject_nme'] = df.subject.map(sub_dict)
    df['sub_hemi'] = df['subject'] + '_' + df['ch_hemisphere']


def _get_neumann_days_afer_surgery(df):
    """For Neumann data, calculate days after surgery."""
    if 'Neumann' not in df.project.unique():
        return None
    for cond in ['off', 'on']:
        mask = (df.project == 'Neumann') & (df.cond == cond)
        df_neu = df[mask]
        col_surgery = 'patient_date_of_implantation'
        col_recording = f'patient_recording_date_{cond.upper()}'
        day_surgery = df_neu[col_surgery]
        day_recording = df_neu[col_recording]
        # Important: UPDRS_Berlin.xlsx has some dates in the wrong format. I
        # had to fix them manually. This will cause difficulties whenever I
        # download the new version online. Therefore, I should always keep a
        # copy of the old corrected file. Notably, subjects EL029-EL031 and
        # L003 + L020-L029 are in the wrong format where day and month might be
        # switched. Do not use dayfirst=True in pd.to_datetime because it
        # avoids errors while incorporating wrong dates. This yields wrong
        # days-after-surgery data. Note that all UPDRS scores and recording
        # dates in the session.tsv files are correct.
        day_recording = pd.to_datetime(day_recording, format='mixed')
        days = day_surgery - day_recording
        days = -days.dt.days
        if not days.dropna().empty:
            # Only check if data is available
            assert days.max() < 10, "Check days after surgery for Neumann."
        df.loc[mask, 'patient_days_after_implantation'] = days
        df.drop(columns=[col_recording], inplace=True)
    df.drop(columns=[col_surgery], inplace=True)
    # Subject L009 had 2 surgeries (see UPDRS Sheet)
    df.loc[(df.subject == 'NeuL009'), 'patient_days_after_implantation'] = 3
    # Dates of surgery and recording possibly flipped?
    df.loc[(df.subject == 'NeuEL026'), 'patient_days_after_implantation'] = 6


def _set_bad_chs_nan(df):
    """Set the values of all bad channels to nan except for arrays.

    This prevents accidentally using them while preserving bad power spectra
    for plotting."""
    # get all columns that are float type
    cols_set_nan = set(df.select_dtypes(include='number').columns)
    updrs_cols = {col for col in cols_set_nan if 'UPDRS' in col}
    patient_cols = {col for col in cols_set_nan if 'patient' in col}
    keep_cols = {'Subject', 'plot_column', 'mni_x', 'mni_y', 'mni_z',
                 'sweet_spot_distance', 'fm_r_squared', 'fm_error',
                 'fm_freq_res', 'Patients'}
    keep_cols = keep_cols | updrs_cols | patient_cols
    cols_set_nan = list(cols_set_nan - keep_cols)

    # find all bad channels and set values of ch_columns to nan
    df.loc[df.ch_bad, cols_set_nan] = np.nan


def _correct_sex(df):
    df.patient_sex.fillna('unknown', inplace=True)
    sex_dic = {'m': 'male', 'M': 'male', 'f': 'female', 'F': 'female',
               'male': 'male', 'female': 'female', 'unknown': 'unknown'}
    df['patient_sex'] = df.patient_sex.apply(sex_dic.get)


def first_value(series):
    if series.isnull().all():
        return np.nan
    else:
        mode_value = series.mode().iloc[0]
        if series.nunique(dropna=False) != 1:
            discrepancy = series[series != mode_value].unique()
            msg = (f"Values are not equal in series: {series.name}.\n"
                   f"Most common value: {mode_value}.\n"
                   f"Discrepancy caused by values: {discrepancy}")
            raise AssertionError(msg)
        return mode_value


def _channel_choice(df):
    """Select channel choices based on various scientific considerations.

    ELECTRODE BASED CHANNEL CHOICES
        Pro: Simple, no MNI needed.
        Contra: Ignores mni localization.
        Skipped in this function because channels can be selected very simple
        for example using
            >> ch_mask = (df.ch == 'LFP_1-3')

    LOCALIZATION BASED CHANNEL CHOICES
        Pro: anatomically defined, therefore more meaningful.
        Contra: Oxford and Hirschmann MNI unknown, needs to be guessed.

    ELECTROPHYSIOLOGY BASED CHANNEL CHOICES
        Pro: subject-specific electrophysiology potentially more accurate than
             subject-unspecific sweet spot?
        Contra: scientific bias towards assuming that beta indicates most
                important channel
    """
    # Initialize columns
    for ch_choice in CH_CHOICES:
        df[ch_choice] = False

    # Choose single amplifier channel
    # For directional DBS Lead datasets: Choose Amp_2-3 which consists of
    # directional fused bipolar channels:
    ring_projects = df.project.isin(['Hirschmann', 'Tan'])
    amp_dir = (df.ch == 'Amp_2-3') & ~ring_projects
    # For nondirectional DBS Lead datasets: Choose Amp_1-4 which consists of
    # nondirectional bipolar channels:
    amp_ring = (df.ch == 'Amp_1-4') & ring_projects
    mask = amp_dir | amp_ring
    df.loc[mask, 'ch_choice_amp'] = True

    dist_chs = df.ch.isin(["LFP_1-3", "LFP_2-4"])
    adj_chs = df.ch.isin(['LFP_1-2', 'LFP_2-3', 'LFP_3-4'])

    # Always use distant sweet spot chs inside STN and replace with most max
    # high beta if unknown.
    # Bad: subject is lost if sweetspot channel is bad or outside STN.
    # Good: high specificity.
    mask_mni = (df.ch_sweetspot & df.ch_inside_stn & ~df.ch_bad)
    mask_beta = (df.mni_x.isna() & df.ch_chmax_beta_high_fm_powers_max_log_off
                 & ~df.ch_bad)
    df.loc[mask_mni & dist_chs, 'ch_dist_sweet_inside'] = True  # MNI known
    df.loc[mask_beta & dist_chs, "ch_dist_sweet_inside"] = True  # MNI unknown
    # Adjacent (without Tan)
    df.loc[mask_mni & adj_chs, 'ch_adj_sweet_inside'] = True  # MNI known
    df.loc[mask_beta & adj_chs, "ch_adj_sweet_inside"] = True  # MNI unknown

    # Same without excluding channels outside STN
    # Pro: Could enable aDBS even with non-optimally placed electrodes.
    mask_mni = (df.ch_sweetspot & ~df.ch_bad)
    df.loc[mask_mni & dist_chs, 'ch_dist_sweet'] = True  # MNI known
    df.loc[mask_beta & dist_chs, "ch_dist_sweet"] = True  # MNI unknown
    df.loc[mask_mni & adj_chs, 'ch_adj_sweet'] = True  # MNI known
    df.loc[mask_beta & adj_chs, "ch_adj_sweet"] = True  # MNI unknown

    # Same but using arithmetic mean of distant sweetspot channels instead of
    # single one.
    mask_mni = (df.ch_sweetspot_dist & ~df.ch_bad)
    df.loc[mask_mni & dist_chs, 'ch_dist_sweet_mean'] = True  # MNI known
    df.loc[mask_beta & dist_chs, "ch_dist_sweet_mean"] = True  # MNI unknown
    df.loc[mask_mni & adj_chs, 'ch_adj_sweet_mean'] = True  # MNI known
    df.loc[mask_beta & adj_chs, "ch_adj_sweet_mean"] = True  # MNI unknown


    # Choose distant channel with highest beta power in OFF condition.
    mask_beta = (df.ch_chmax_beta_high_fm_powers_max_log_off
                 & ~df.ch_bad)
    df.loc[mask_beta & dist_chs, "ch_dist_beta_high_max_off"] = True
    df.loc[mask_beta & adj_chs, "ch_adj_beta_high_max_off"] = True

    # Testing and correcting datatype
    no_duplicates = ["subject", "ch_hemisphere", "fm_params",
                     "psd_kind", 'ch_bip_distant', 'cond']
    for ch_choice in CH_CHOICES:
        df[ch_choice] = df[ch_choice].astype(bool)
        group = df[df[ch_choice]].groupby(no_duplicates)
        assert (group.ch.nunique() <= 1).all(), ch_choice
        df_small = df[(df.fm_params == 'broad') & (df.psd_kind == 'standard')
                      & (df.cond == 'off')]
        print(f"Chosen channels for {ch_choice}: {df_small[ch_choice].sum()}")


def _same_amp_cond(df):
    """Add mask to easily remove patients who had different amplifiers
    or recording sites for ON and OFF conditions."""
    df['cond_same_amp'] = False
    # collect subjects with the same amplifier in ON and OFF condition
    sub_single_amp = df.groupby('subject')['amplifier'].nunique() == 1
    subs_single_amp = sub_single_amp[sub_single_amp].index
    df.loc[df.subject.isin(subs_single_amp), 'cond_same_amp'] = True
    amp_subs = ['NeuEmptyroom', 'TanEmptyroom', 'HirEmptyroom', 'FloEmptyroom']
    df.loc[df.subject.isin(amp_subs), 'cond_same_amp'] = True


def _select_max_band(df, exclude_bad=True, bands=None):
    """Select maximum beta power of distant ring and adjacent channels.

    Exclude bad channels can lead to bad choices. However, better than
    including bad channels which would also lead to bad choice."""
    no_duplicates_total = ["subject", "ch_hemisphere",
                           "fm_params",
                           "psd_kind",
                           'ch_bip_distant']
    no_duplicates_periodic = ["subject", "ch_hemisphere", 'ch_bip_distant']
    no_duplicates_total_cond = no_duplicates_total.copy() + ['cond']
    no_duplicates_periodic_cond = no_duplicates_periodic.copy() + ['cond']

    dist_chs = df.ch.isin(["LFP_1-3", "LFP_2-4"])
    adj_chs = df.ch.isin(['LFP_1-2', 'LFP_2-3', 'LFP_3-4'])
    bip_chs = dist_chs | adj_chs

    bands = cfg.BANDS.keys() if bands is None else bands
    band_powers = ['_abs_max_log', '_abs_mean_log', '_fm_powers_max_log']

    for band in bands:
        for band_power in band_powers:
            if band == 'FTG' and 'abs' in band_power:
                continue
            col_pwr = band + band_power
            col_ch = f"ch_chmax_{col_pwr}"
            df[col_ch] = False

            bads = ~df.ch_bad if exclude_bad else slice(None)
            mask = (bads & bip_chs)
            if 'fm' in col_pwr:
                no_duplicates = no_duplicates_periodic
                no_duplicates_cond = no_duplicates_periodic_cond
            else:
                no_duplicates = no_duplicates_total
                no_duplicates_cond = no_duplicates_total_cond

            group = df[mask].groupby(no_duplicates_cond)

            def custom_idxmax(sub_df, col_pwr):
                if col_pwr.endswith('_fm_powers_max_log'):
                    # only consider fm_params = 'broad' and forward to other
                    # fm_params. Important difference to total power where we
                    # want ch_maxima for normalized and absolute power
                    # separately.
                    fm_params = 'broad'
                    if 'FTG' in col_pwr:
                        fm_params = 'gamma'
                        not_FTG = False
                    else:
                        not_FTG = True

                    sub_df_broad = sub_df[sub_df.fm_params == fm_params]
                    # for periodic fm_powers, only consider non-zero peaks.
                    all_zero = (sub_df_broad[col_pwr].dropna() == 0).all()
                    all_nan = sub_df_broad[col_pwr].isna().all()
                    if (all_zero or all_nan) and not_FTG:
                        # if all peaks are zero, use total power
                        col_pwr = col_pwr.replace('_fm_powers_max_log',
                                                  '_abs_max_log')
                        group_total = sub_df.groupby(['fm_params', 'psd_kind'])
                        return group_total[col_pwr].idxmax().to_numpy()
                    else:
                        ch_idx = sub_df_broad[col_pwr].idxmax()
                        ch_max = sub_df_broad.loc[ch_idx, 'ch_nme']
                        ch_idcs = sub_df.ch_nme.isin([ch_max])
                        ch_idcs = ch_idcs[ch_idcs].index.to_numpy()
                        return ch_idcs
                else:
                    return sub_df[col_pwr].idxmax()

            apply_idx = lambda sub_df: custom_idxmax(sub_df, col_pwr)
            ch_band_max = group.apply(apply_idx)
            if 'fm' in col_pwr:
                ch_band_max = np.hstack(ch_band_max.to_numpy()).flatten()
            df.loc[ch_band_max, col_ch] = True

            for cond in ['off', 'on']:
                col_ch_cond = col_ch + f"_{cond}"
                df[col_ch_cond] = False

                cond_mask = (df.cond == cond)
                group = df[mask & cond_mask].groupby(no_duplicates)
                ch_band_max = group.apply(apply_idx)
                if ch_band_max.empty:
                    # channel might be bad or missing in other condition
                    continue
                if 'fm' in col_pwr:
                    ch_band_max = np.hstack(ch_band_max.to_numpy()).flatten()
                df.loc[ch_band_max, col_ch_cond] = True

                # Propagate the selected channels to the other condition
                def cond_idxmax(sub_df, col_ch_cond, cond):
                    if sub_df[sub_df.cond == cond].empty:
                        # if opposite condition is empty, choose channel of
                        # current condition. This increases the sample size
                        # and is more accurate than skipping the hemisphere.

                        # col_ch_cond = col_ch_cond.replace(f"_{cond}", "")

                        # Later Moritz disagrees. If opposite condition is
                        # empty, e.g. due to bad channels, it does not make
                        # sense to choose some channels randomly and to have
                        # more channels in one condition than in the other.
                        return None
                    ch_mask = sub_df[col_ch_cond] == True
                    ch_max = sub_df.loc[ch_mask, 'ch_nme'].unique()
                    # find channel index in opposite condition
                    other_cond = 'off' if cond == 'on' else 'on'
                    sub_df_cond = sub_df[(sub_df.cond == other_cond)]
                    ch_idx = sub_df_cond.ch_nme.isin(ch_max)
                    # skip if ch_nme only in current condition present
                    if ch_idx[ch_idx].empty:
                        return None
                    if '_fm_powers_' in col_ch_cond:
                        # return multiple indices
                        return ch_idx[ch_idx].index.to_numpy()
                    else:
                        return ch_idx[ch_idx].index.to_numpy()[0]

                apply_cond = lambda sub_df: cond_idxmax(sub_df, col_ch_cond,
                                                        cond)
                # use no_duplicates_total to correctly select channels also
                # by fm_params and psd_kind
                group = df[mask].groupby(no_duplicates_total)
                ch_band_max = group.apply(apply_cond).dropna()
                if ch_band_max.empty:
                    # channel might be bad or missing in other condition
                    continue
                if 'fm' in col_pwr:
                    ch_band_max = np.hstack(ch_band_max.to_numpy()).flatten()
                df.loc[ch_band_max, col_ch_cond] = True


def _updrs_add_bradyrigid(df):
    """Combine bradykinesia and rigditiy UPDRS subscores to bradyrigid.

    Also add total bradyrigid scores for both hemispheres."""
    for prepost in ["_pre", "_post"]:
        # Add bradyrigid scores left and right
        for project in df.project.unique():
            mask = (df.project == project)
            df_project = df[mask]
            col = f"UPDRS{prepost}_bradyrigid_left"
            no_bradyrigid = (col not in df_project.columns
                             or df_project[col].isna().all())
            if no_bradyrigid:
                col = f"UPDRS{prepost}_bradykinesia_left"
                if col not in df_project.columns:
                    # pre UPDRS might not exist
                    continue
                brady_left = df_project[f'UPDRS{prepost}_bradykinesia_left']
                rigid_left = df_project[f'UPDRS{prepost}_rigidity_left']
                brady_right = df_project[f'UPDRS{prepost}_bradykinesia_right']
                rigid_right = df_project[f'UPDRS{prepost}_rigidity_right']
                bradyrigid_left = (brady_left + rigid_left)
                bradyrigid_right = (brady_right + rigid_right)
                both_left = f"UPDRS{prepost}_bradyrigid_left"
                both_right = f"UPDRS{prepost}_bradyrigid_right"
                df.loc[mask, both_left] = bradyrigid_left
                df.loc[mask, both_right] = bradyrigid_right

        # Add total scores
        for project in df.project.unique():
            mask = (df.project == project)
            df_project = df[mask]
            for score in ["bradyrigid", "bradykinesia", 'rigidity', 'tremor']:
                col = f"UPDRS{prepost}_{score}_total"
                no_total_score = (col not in df_project.columns
                                  or df_project[col].isna().all())
                # Florin and Neumann have total scores already
                if no_total_score:
                    # make total bradyrigid score from total bradykinesia
                    # and rigidity scores. More precise than using left
                    # and right scores.
                    col_brady = col.replace(score, "bradykinesia")
                    col_rigid = col.replace(score, "rigidity")
                    if (col_brady in df_project.columns
                            and len(df_project[col_brady].dropna())
                            and score == "bradyrigid"):
                        score_brady = df_project[col_brady]
                        score_rigid = df_project[col_rigid]
                        assert col_rigid in df_project.columns
                        score_total = score_brady + score_rigid
                        df.loc[mask, col] = score_total
                    else:
                        # If no total scores, make from left and right hemi
                        col_left = f"UPDRS{prepost}_{score}_left"
                        col_right = f"UPDRS{prepost}_{score}_right"
                        if col_left not in df_project.columns:
                            # pre UPDRS might not exist
                            continue
                        score_left = df_project[col_left]
                        score_right = df_project[col_right]
                        score_total = score_left + score_right
                        df.loc[mask, col] = score_total
    return df


def _updrs_ipsi_contra(df):
    """Ipsi vs contra hemi for handedness, symptom dominant side, and UPDRS."""
    # Change labels to L and R
    lr_dic = dict(left="L", right="R")
    df.replace({"patient_handedness": lr_dic}, inplace=True)
    df.replace({"patient_symptom_dominant_side": lr_dic}, inplace=True)
    df.replace({"ECOG_hemisphere": lr_dic}, inplace=True)
    try:
        df.patient_handedness.fillna('unknown', inplace=True)
        df.patient_symptom_dominant_side.fillna('unknown', inplace=True)
    except AttributeError:
        pass

    # Assign left and right to contra and ipsi
    updrs_cols = ["UPDRS_pre_bradykinesia",
                  "UPDRS_pre_rigidity",
                  "UPDRS_pre_tremor",
                  "UPDRS_pre_bradyrigid",

                  "UPDRS_post_bradykinesia",
                  "UPDRS_post_rigidity",
                  "UPDRS_post_tremor",
                  "UPDRS_post_bradyrigid",
                  ]
    lr_opposite = dict(left="R", right="L")
    for col in updrs_cols:
        for key, val in lr_dic.items():
            idx_ipsi = (df.ch_hemisphere == val, f"{col}_ipsi")
            idx_contra = (df.ch_hemisphere == lr_opposite[key],
                          f"{col}_contra")
            try:
                df.loc[idx_ipsi] = df[f"{col}_{key}"]
                df.loc[idx_contra] = df[f"{col}_{key}"]
            except KeyError:
                # Hirschmann has only bradyrigid columns
                pass
    return df


def _symptom_dominant_side(df):
    # Calc symptom dominant side according to bradykinesia-rigidity subscores
    contra_stronger = df.UPDRS_bradyrigid_contra > df.UPDRS_bradyrigid_ipsi
    ipsi_stronger = df.UPDRS_bradyrigid_contra < df.UPDRS_bradyrigid_ipsi
    equal = df.UPDRS_bradyrigid_contra == df.UPDRS_bradyrigid_ipsi
    df.loc[contra_stronger, "patient_symptom_dominant_side_BR"] = "severe side"
    df.loc[ipsi_stronger, "patient_symptom_dominant_side_BR"] = "mild side"
    df.loc[equal, "patient_symptom_dominant_side_BR"] = "equal"

    # Calc symptom dominant side by condition
    for cond in ['off', 'on']:
        colBR = f"patient_symptom_dominant_side_BR_{cond}"
        mask = (df.cond == cond)

        # Recalculate bradykinesia-rigidity dominance for each condition
        con = 'UPDRS_bradyrigid_contra'
        ipsi = 'UPDRS_bradyrigid_ipsi'
        contra_stronger = df.loc[mask, con] > df.loc[mask, ipsi]
        ipsi_stronger = df.loc[mask, con] < df.loc[mask, ipsi]
        equal = df.loc[mask, con] == df.loc[mask, ipsi]

        # Assign dominant side based on condition and calculated values
        df.loc[mask & contra_stronger, colBR] = "severe side"
        df.loc[mask & ipsi_stronger, colBR] = "mild side"
        df.loc[mask & equal, colBR] = "equal"

    def fill_missing_sides(group):
        # Fill missing 'on' values with 'off' values, and vice versa
        group[col_off] = group[col_off].ffill().bfill()
        group[col_on] = group[col_on].ffill().bfill()
        return group

    # Ensure the correct values are assigned for both 'on' and 'off' conditions
    col_off = 'patient_symptom_dominant_side_BR_off'
    col_on = 'patient_symptom_dominant_side_BR_on'
    # Apply the function per subject and hemisphere
    df = df.groupby(['subject', 'ch_hemisphere']).apply(fill_missing_sides)
    df = df.reset_index(drop=True)

    # Testing
    subset = ['UPDRS_bradyrigid_contra', 'UPDRS_bradyrigid_ipsi']
    df_off = df[df.cond == 'off'].dropna(subset=subset)
    df_on = df[df.cond == 'on'].dropna(subset=subset)
    assert np.all(df_off.patient_symptom_dominant_side_BR
                  == df_off.patient_symptom_dominant_side_BR_off)
    assert np.all(df_on.patient_symptom_dominant_side_BR
                  == df_on.patient_symptom_dominant_side_BR_on)

    # Add column 'dominant_side_OffvsOn' based on comparison between 'off' and
    # 'on' conditions. Value is False if one condition is missing.
    df.loc[:, 'dominant_side_consistent'] = df[col_off] == df[col_on]
    # Value is False if no dominant side in off or on state
    equal = (df[col_off] == 'equal') | (df[col_on] == 'equal')
    df.loc[equal, 'dominant_side_consistent'] = False
    return df


def _fill_missing_updrs(df):
    """Fill in missing pre-op and post-op UPDRS scores with post-op and pre-op
    respectively."""
    updrs_cols = ["UPDRS_III",
                  "UPDRS_bradykinesia_contra",
                  "UPDRS_bradykinesia_ipsi",
                  "UPDRS_rigidity_contra",
                  "UPDRS_rigidity_ipsi",
                  "UPDRS_bradyrigid_contra",
                  "UPDRS_bradyrigid_ipsi",
                  "UPDRS_tremor_contra",
                  "UPDRS_tremor_ipsi"]

    for col in updrs_cols:
        col_pre = col.replace("UPDRS_", "UPDRS_pre_")
        col_post = col.replace("UPDRS_", "UPDRS_post_")
        # use post as a default and fill with pre if missing
        df.loc[:, col] = df[col_post]
        if col_post and col_pre in df.columns:
            mask = df[col_post].isna() & df[col_pre].notna()
            df.loc[mask, col] = df.loc[mask, col_pre]


def _brady_severity(df, updrs_kinds=['UPDRS_bradyrigid_contra', 'UPDRS_III']):
    """Calculate bradykinesia severity categories."""
    assert 'all' not in df.project.unique(), "Calc quartiles project basesd"
    if df.subject.nunique() < 3:
        return df

    def calculate_categories(df, updrs_kind):
        labels_median = ['mild_half', 'severe_half']
        labels_thirds = ['lower_third', 'moderate_third', 'upper_third']
        labels_quantiles = ['Q1', 'Q2', 'Q3', 'Q4']
        halves = pd.qcut(df[updrs_kind], q=2, labels=labels_median)
        thirds = pd.qcut(df[updrs_kind], q=3, labels=labels_thirds)
        quants = pd.qcut(df[updrs_kind], q=4, labels=labels_quantiles)
        df[f'{updrs_kind}_severity_median'] = halves
        df[f'{updrs_kind}_severity_thirds'] = thirds
        df[f'{updrs_kind}_severity_quartiles'] = quants
        return df.drop(columns=col)  # drop for later merge


    no_duplicates = ['project', 'subject', 'cond']
    for updrs_kind in updrs_kinds:
        col = updrs_kind
        # only keep required columns for subframe
        if 'contra' in updrs_kind:
            add_hemi = ['ch_hemisphere']
        else:
            add_hemi = []
        keep_cols = no_duplicates + add_hemi + [col]
        # get slim df
        df_unique = df.loc[df[col].notna(), keep_cols]
        if not len(df_unique):
            continue
        df_unique = df_unique.drop_duplicates(subset=no_duplicates + add_hemi)
        # group by condition
        # important: group by project to prohobit dataset bias
        group = df_unique.groupby(['project', 'cond'], group_keys=False)
        # apply
        df_severity = group.apply(calculate_categories,
                                  updrs_kind=updrs_kind)
        # merge back to df
        df = df.merge(df_severity, on=no_duplicates + add_hemi, how='left')

    # testing
    msg = 'Counts dont match'
    for project in df.project.unique():
        for cond in ['off', 'on', 'offon_abs']:
            mask = (df.cond == cond) & (df.project == project)
            df_test = df[mask].drop_duplicates(subset=no_duplicates
                                               + ['ch_hemisphere'])
            median = df_test.UPDRS_bradyrigid_contra.median()
            quartiles = df_test.UPDRS_bradyrigid_contra_severity_quartiles
            vals = quartiles.value_counts()
            vals = vals.sort_index()
            n_below_median = (df_test.UPDRS_bradyrigid_contra <= median).sum()
            n_above_median = (df_test.UPDRS_bradyrigid_contra > median).sum()
            assert vals[0] + vals[1] == n_below_median, msg
            assert vals[2] + vals[3] == n_above_median, msg
    return df


def _correct_types(df):
    """Correct data types."""
    boolean_columns = ["ch_bad", "ch_bip_distant", "ch_directional",
                       'ch_combined_ring',
                       'ch_beta_max', 'ch_sweetspot',
                       'ch_inside_stn', 'ch_wiestpick', 'ch_mean_inside_stn',
                       'ch_mean', 'ch_inside_stn_mean', 'DBS_directional'
                       ] + CH_CHOICES
    ch_max_cols = [col for col in df.columns if col.startswith('ch_chmax')]
    boolean_columns += ch_max_cols
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    float_columns = ["patient_age", "patient_disease_duration"]
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)


def _sweet_spot_distance(df):
    """Calculate distance to sweet spot for all LFP channels
    (also directional and distant bipolar pairs)."""
    # set 0 0 0 coordinates to NaN
    all_zero = (df.mni_x == 0) & (df.mni_y == 0) & (df.mni_z == 0)
    df.loc[all_zero, "mni_x"] = np.nan
    df.loc[all_zero, "mni_y"] = np.nan
    df.loc[all_zero, "mni_z"] = np.nan
    df.loc[all_zero, "mni_coords"] = np.nan

    # Dembek, et al. "Probabilistic Sweet Spots Predict Motor Outcome for Deep
    # Brain Stimulation in Parkinson Disease." ANN NEUROL 2019;86:527–538.
    stn_motor_R = np.array([12.5, -12.72, -5.38])
    stn_motor_L = np.array([-12.68, -13.53, -5.38])

    cond_L = (df.ch_hemisphere == "L") & (df.ch_type_bids == "lfp")
    cond_R = (df.ch_hemisphere == "R") & (df.ch_type_bids == "lfp")

    df_L = df[cond_L]
    df_R = df[cond_R]

    coords_L = np.array([df_L.mni_x, df_L.mni_y, df_L.mni_z])
    coords_R = np.array([df_R.mni_x, df_R.mni_y, df_R.mni_z])

    dist_L = np.linalg.norm(coords_L.T - stn_motor_L, axis=1)
    dist_R = np.linalg.norm(coords_R.T - stn_motor_R, axis=1)

    df.loc[cond_L, "sweet_spot_distance"] = dist_L
    df.loc[cond_R, "sweet_spot_distance"] = dist_R


def _sweetest_adjacent_chs(df):
    """Find LFP contacts closest to the sweet spot for each subject.

    Attention: Distant sweetspot channels are selected in _03_make_dataframe.py
    to enable closest single channel distance. Here, for adjacent channels,
    the arithmetic mean is selected."""
    # Get closest STN contact for each subject, hemisphere, reference,
    # and condition. Distance measured from arithmetic mean of bipolar chs.
    no_duplicates = ["subject", "ch_hemisphere", "cond", "fm_params",
                     "psd_kind"]

    # # Initialize column
    df["ch_sweetspot_dist"] = False

    # Bipolar adjacent contacts
    adj_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    dist_chs = ['LFP_1-3', 'LFP_2-4']
    mask_adj = df.ch.isin(adj_chs) & df.sweet_spot_distance.notna()
    mask_dist = df.ch.isin(dist_chs) & df.sweet_spot_distance.notna()

    group_adj = df[mask_adj].groupby(no_duplicates)
    group_dist = df[mask_dist].groupby(no_duplicates)
    sweetest_rings_adj = group_adj.sweet_spot_distance.idxmin()
    sweetest_rings_dist = group_dist.sweet_spot_distance.idxmin()

    df.loc[sweetest_rings_adj, "ch_sweetspot"] = True
    df.loc[sweetest_rings_dist, "ch_sweetspot_dist"] = True

    df['ch_sweetspot'] = df['ch_sweetspot'].astype(bool)
    df['ch_sweetspot_dist'] = df['ch_sweetspot_dist'].astype(bool)


def _add_ch_colors(df):
    """Add plotting colors for channels."""
    if df.empty:
        raise ValueError("No BIDS paths found.")
    plot_colors = cfg.CHANNEL_PLOT_COLORS
    plot_colors_long = cfg.CHANNEL_PLOT_COLORS_LONG

    df["plot_color"] = df["ch_nme"].map(plot_colors)
    plot_colors_long = df["ch_nme"].map(plot_colors_long)
    df.loc[df.subject == "NeuEL008", "plot_color"] = plot_colors_long
    df.loc[df.subject == "HirML003", "plot_color"] = plot_colors_long
    df.loc[df.subject == "HirML008", "plot_color"] = plot_colors_long
    df.loc[df.ch_bad, "plot_color"] = "grey"
    df.loc[df.plot_color.isna(), "plot_color"] = "darkgrey"


def _add_psd_min_max(df):
    """Apply min and max to PSDs combined for the on and off condition
    (because they are similar) but separately for the offon difference which
    has a much smaller PSD scale.

    Used later for plotting."""
    for conds in [["on", "off"], ["offon_abs"]]:
        mask_conds = (df.cond.isin(conds))
        for spect in ["psd", "asd"]:
            array = np.stack(df.loc[mask_conds, spect].dropna().values)
            mask = mask_conds & df[spect].notna()
            for func in [np.nanmin, np.nanmax]:
                func_str = func.__name__.replace('nan', '')
                col = f"{spect}_{func_str}"
                df.loc[mask, col] = func(array, 1)


def uniform_value_or_null(series):
    """Check all elements in aggregation and return string if all equal."""
    try:
        all_equal = series.nunique(dropna=True) == 1
    except TypeError:
        # lists and ndarrays don't work
        return np.nan

    if all_equal:
        return series.dropna().iloc[0]
    else:
        return np.nan


def _add_ch_means(df):
    """Add mean values for all LFP channels and inside STN channels.

    Return boolean or string if all values are equal, otherwise NaN. Return
    NaN for lists and ndarrays."""
    df['ch_mean'] = False
    df['ch_inside_stn_mean'] = False

    no_duplicates = ["subject", "ch_hemisphere", "cond", "fm_params",
                     "psd_kind"]
    bip_chs = ['LFP_1-2', 'LFP_2-3', 'LFP_3-4']
    dist_chs = ['LFP_1-3', 'LFP_2-4']
    chs = bip_chs + dist_chs  # include all channels to maximize averaging
    mask = ((df.ch_type_bids == "lfp") & (df.ch_reference == 'bipolar')
            # include bad chs because set to nan anyways
            & ~df.ch_mean_inside_stn
            & df.ch.isin(chs))
    mask_dist = ((df.ch_type_bids == "lfp") & (df.ch_reference == 'bipolar')
                 # include bad chs because set to nan anyways
                 & ~df.ch_mean_inside_stn
                 & df.ch.isin(dist_chs))

    # Define the aggregation dictionary and apply
    agg_columns = [col for col in df.columns if col not in no_duplicates]
    number_cols = df[agg_columns].select_dtypes(include='number').columns
    agg_dict = {col: np.nanmean for col in number_cols}
    agg_dict.update({col: uniform_value_or_null for col in agg_columns
                     if col not in agg_dict})
    group_all = df[mask].groupby(no_duplicates)
    group_all_dist = df[mask_dist].groupby(no_duplicates)
    group_stn = df[mask & df.ch_inside_stn].groupby(no_duplicates)

    ch_all_mean = group_all.agg(agg_dict).reset_index()
    ch_all_mean_dist = group_all_dist.agg(agg_dict).reset_index()
    ch_stn_mean = group_stn.agg(agg_dict).reset_index()

    # all chs of hemisphere are bad: ch_bad = False correctly set
    # all chs of hemisphere are good: ch_bad = True correctly set
    # some chs of hemisphere are good: ch_bad = np.nan. Set to False, since
    # bad chs were ignored during averaging:
    ch_all_mean['ch_bad'] = ch_all_mean['ch_bad'].fillna(False)
    ch_all_mean_dist['ch_bad'] = ch_all_mean_dist['ch_bad'].fillna(False)
    ch_stn_mean['ch_bad'] = ch_stn_mean['ch_bad'].fillna(False)

    # Add manually
    ch_all_mean['ch'] = 'LFP_mean'
    hemi_str = lambda hemi: f'LFP_{hemi}_mean'
    ch_all_mean['ch_nme'] = ch_all_mean['ch_hemisphere'].apply(hemi_str)
    ch_all_mean['ch_mean'] = True
    ch_all_mean['ch_inside_stn_mean'] = False
    ch_all_mean['ch_beta_max'] = False
    ch_all_mean['ch_sweetspot'] = False
    ch_all_mean['ch_bip_distant'] = True
    ch_all_mean['ch_combined_ring'] = True
    ch_chmax_cols = [col for col in df.columns if col.startswith('ch_chmax')]
    for col in ch_chmax_cols:
        ch_all_mean[col] = False
    for col in CH_CHOICES:
        ch_all_mean[col] = False

    # Add manually
    ch_all_mean_dist['ch'] = 'LFP_mean_dist'
    hemi_str = lambda hemi: f'LFP_{hemi}_meandist'
    ch_all_mean_dist['ch_nme'] = ch_all_mean_dist['ch_hemisphere'].apply(hemi_str)
    ch_all_mean_dist['ch_mean'] = True
    ch_all_mean_dist['ch_inside_stn_mean'] = False
    ch_all_mean_dist['ch_beta_max'] = False
    ch_all_mean_dist['ch_sweetspot'] = False
    ch_all_mean_dist['ch_bip_distant'] = True
    ch_all_mean_dist['ch_combined_ring'] = True
    ch_chmax_cols = [col for col in df.columns if col.startswith('ch_chmax')]
    for col in ch_chmax_cols:
        ch_all_mean_dist[col] = False
    for col in CH_CHOICES:
        ch_all_mean_dist[col] = False

    ch_stn_mean['ch'] = 'LFP_STNmean'
    hemi_str = lambda hemi: f'LFP_{hemi}_STNmean'
    ch_stn_mean['ch_nme'] = ch_stn_mean['ch_hemisphere'].apply(hemi_str)
    ch_stn_mean['ch_inside_stn_mean'] = True
    ch_stn_mean['ch_mean'] = False
    ch_stn_mean['ch_beta_max'] = False
    ch_stn_mean['ch_sweetspot'] = False
    ch_stn_mean['ch_bip_distant'] = True
    ch_stn_mean['ch_combined_ring'] = True
    for col in ch_chmax_cols:
        ch_stn_mean[col] = False
    for col in CH_CHOICES:
        ch_stn_mean[col] = False

    # Append the new rows to the original DataFrame
    dfs = [df, ch_all_mean, ch_stn_mean, ch_all_mean_dist]
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


def _add_psd_means(df):
    # dropna important for offon_rel psd column which is nan
    resolutions = df.fm_freq_res.dropna().unique()
    assert len(resolutions) == 1, "Multiple frequency resolutions found."
    freqs = df.psd_freqs.iloc[0]
    psd_arr = np.stack(df.psd.values)
    psd_arr_log = np.log10(psd_arr)
    df["psd_mean_log"] = np.nanmean(psd_arr_log, 1)

    frequency_masks = [(1, 100), (5, 95), (100, 200), (150, 200), (100, 150),
                       (70, 120), (25, 95)]

    for f_low, f_high in frequency_masks:
        mask_freqs = (freqs >= f_low) & (freqs <= f_high)
        col = f"psd_mean_{f_low}to{f_high}_log"
        df[col] = np.nanmean(psd_arr_log[:, mask_freqs], 1)

    # Add normalization range as linear sum
    mask_freqs = (freqs >= 5) & (freqs <= 95)
    df['psd_sum_5to95'] = np.nansum(psd_arr[:, mask_freqs], 1)

    f_low, f_high = (2, 45)
    avoid_beta = ((freqs < cfg.BANDS["beta"][0])
                  | (freqs > cfg.BANDS["beta"][1]))
    mask_freqs = (freqs >= f_low) & (freqs <= f_high) & avoid_beta
    col = f"psd_mean_{f_low}to{f_high}_noBeta_log"
    df[col] = np.nanmean(psd_arr_log[:, mask_freqs], 1)

    avoid_alpha_beta = ((freqs < cfg.BANDS["alpha_beta"][0])
                        | (freqs > cfg.BANDS["alpha_beta"][1]))
    mask_freqs = (freqs >= f_low) & (freqs <= f_high) & avoid_alpha_beta
    col = f"psd_mean_{f_low}to{f_high}_noAlphaBeta_log"
    df[col] = np.nanmean(psd_arr_log[:, mask_freqs], 1)

    f_low, f_high = (1, 95)
    avoid_alpha_beta = ((freqs < cfg.BANDS["alpha_beta"][0])
                        | (freqs > cfg.BANDS["alpha_beta"][1]))
    mask_freqs = (freqs >= f_low) & (freqs <= f_high) & avoid_alpha_beta
    col = f"psd_mean_{f_low}to{f_high}_noAlphaBeta_log"
    df[col] = np.nanmean(psd_arr_log[:, mask_freqs], 1)


def _absolute_band_power(df, bands=None):
    """Add PSD mean and max and min values for each frequency band."""
    resolutions = df.fm_freq_res.dropna().unique()
    assert len(resolutions) == 1, "Multiple frequency resolutions found."
    resolution = resolutions[0]
    bands = cfg.BANDS.keys() if bands is None else bands
    freqs = df.psd_freqs.iloc[0]
    psd_arr = np.stack(df.psd.values)
    psd_arr_log = np.log10(psd_arr)
    for band in bands:
        if band == 'FTG':
            # added later
            continue
        (f_low, f_high) = cfg.BANDS[band]
        mask = (freqs >= f_low) & (freqs < f_high)

        # Add absolute band power linear and log
        for func in [np.nanmax, np.nanmean, np.nanmin]:

            func_str = func.__name__.replace('nan', '')  # -> "nanmax" -> "max"
            col_nme = f"{band}_abs_{func_str}"
            df[col_nme] = func(psd_arr[:, mask], 1)
            col_nme = f"{band}_abs_{func_str}_log"
            df[col_nme] = func(psd_arr_log[:, mask], 1)

        # Add frequency of maximum absolute power within band
        row_max_freq = f"{band}_abs_max_freq"
        freq_idcs = psd_arr[:, mask].argmax(1)
        df[row_max_freq] = freqs[mask][freq_idcs]

        # Add 5 Hz mean power around peak frequency
        peak_freqs = freqs[mask][freq_idcs]
        window = 2 * resolution
        freq_idcs_mean = ((freqs >= (peak_freqs[:, None] - window))
                          & (freqs <= (peak_freqs[:, None] + window)))
        col_nme = f"{band}_abs_max5Hz"
        df[col_nme] = np.nanmean(np.where(freq_idcs_mean, psd_arr, np.nan),
                                 axis=1)
        col_nme = f"{band}_abs_max5Hz_log"
        df[col_nme] = np.nanmean(np.where(freq_idcs_mean, psd_arr_log, np.nan),
                                 axis=1)

        ### TESTING  ##########################################################
        # import matplotlib.pyplot as plt
        # plt.figure()
        # mask__ = freqs < 30
        # plt.plot(freqs[mask__], psd_arr_log[idx][mask__])
        # plt.plot(df[f'{band}_abs_max_freq'][idx], df[f'{band}_abs_max_log'][idx], 'ro')
        # plt.vlines(f_low, *plt.ylim(), color='orange')
        # plt.vlines(f_high, *plt.ylim(), color='orange')
        # plt.hlines(df[f'{band}_abs_mean_log'][idx], *plt.xlim(), color='r')
        # plt.hlines(df[f'{band}_abs_mean5Hz_log'][idx], peak_freqs[idx]-2, peak_freqs[idx]+2, color='g')
        # plt.pause(1)
        ### TESTING  ##########################################################

        assert df[row_max_freq].max() <= f_high
        assert df[row_max_freq].max() >= f_low
        # Testing: ############################################################
        # check whether the value row_max_freq at psd_freqs is the same
        # as the value row_max for the first index
        idx0 = 0
        freqs_idx0 = df.psd_freqs.values[idx0]
        freqs_max_idx0 = df[row_max_freq].values[idx0]
        fmax_idx0 = np.where(freqs_idx0 == freqs_max_idx0)[0][0]
        psd_idx0 = df.psd.values[idx0]
        band_pwr_max_idx0 = df[f"{band}_abs_max"].values[idx0]
        assert band_pwr_max_idx0 == psd_idx0[fmax_idx0]
        #######################################################################

        # Add alternative band power measures. Need to loop over each row
        # since mask is different for each row.
        for idx, row in df.dropna(subset=row_max_freq).iterrows():
            fmax = row[row_max_freq]
            # for fmean in [1, 3, 10]:
            for fmean in [1, 3]:  # +- 1 Hz and 3 Hz
                mean_low = fmax - fmean
                mean_max = fmax + fmean
                mask = ((row.psd_freqs >= mean_low)
                        & (row.psd_freqs <= mean_max))
                freqs_averaged = fmean * 2 + 1  # both sides + abs_max_freq
                col_nme = f"{band}_abs_max_{freqs_averaged}Hz"
                psd_peak = row.psd[mask]
                df.loc[idx, col_nme] = np.mean(psd_peak)
                col_nme += "_log"
                df.loc[idx, col_nme] = np.mean(np.log10(psd_peak))


def freq_max(df, f_low, f_high):
    pwr = df.psd
    freqs = df.psd_freqs
    mask = (freqs >= f_low) & (freqs <= f_high)
    fmax_band_idx = pwr[mask].argmax()
    freqs_band = freqs[mask]
    fmax = freqs_band[fmax_band_idx]
    return fmax


def _subtract_noise_from_power(df, bands=None):
    """Correct HFOs by subtracting noise floor.

    DOES NOT MAKE SENSE. Noise floor is largely neurphysiological signal."""
    # subtract noise floor minimum from all band powers as a correction
    if bands:
        HFO_bands = [band for band in bands if band.startswith("HFO")]
    else:
        HFO_bands = ['HFO', 'HFO_low', 'HFO_high']
    if not HFO_bands:
        return
    freqs = df.psd_freqs.iloc[0]
    for project in df.project.unique():
        (f_low, f_high) = cfg.NOISE_FLOORS[project]
        mask = (df.project == project)

        psd_arr = np.stack(df[mask].psd.values)
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        col = "noise_floor_abs_min"
        df.loc[mask, col] = psd_arr[:, freq_mask].min(1)
        psd_arr_log = np.log10(psd_arr)
        df.loc[mask, f'{col}_log'] = psd_arr_log[:, freq_mask].min(1)

    noise_floor = df.noise_floor_abs_min
    for band in HFO_bands:
        if band not in df.columns:
            continue
        pwr = "_abs_max"
        col_nme = band + pwr
        col_nonoise = f"{col_nme}_nonoise"
        col_log_nonoise = f"{col_nme}_log_nonoise"

        df[col_nonoise] = df[col_nme] - noise_floor
        df[col_log_nonoise] = np.log10(df[col_nme] - noise_floor)


def band_pwr(df, func, mask=None, scale="linear", col="psd", f_low=None,
             f_high=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ignore_warnings()
        pwr = df[col] if scale == "linear" else np.log10(df[col])
        if f_low is not None and f_high is not None:
            if mask is not None:
                raise ValueError("mask and f_low/f_high ")
            mask = (df.psd_freqs >= f_low) & (df.psd_freqs <= f_high)
        if mask is not None:
            pwr = pwr[mask]
        return func(pwr)


def _exclude_bad_fits(df):
    """Exclude bad fooof fits."""
    for fm_params in df.fm_params.unique():
        # Exclude bad fits based on r^2 and error (broad params)
        try:
            error_dict = cfg.FIT_ERROR_THRESHOLDS[fm_params]
        except KeyError:
            continue
        r_squared_threshold = error_dict['r_squared_threshold']
        mask_params = (df.fm_params == fm_params) & (df.bids_task != 'noise')
        mask_r2 = (df.fm_r_squared < r_squared_threshold)
        exclude = len(df[mask_params & mask_r2])
        include = len(df[mask_params & ~mask_r2])
        print(f"{fm_params}: {exclude / include * 100:.2f}% "
              "excluded due to bad fits")

        # set all fooof results below threshold to nan
        fm_columns = {col for col in df.columns if 'fm' in col}
        keep = {'fm_params', 'fm_r_squared', 'fm_freqs', 'fm_fit_range',
                'fm_freq_res', 'fm_info'}
        cols_set_nan = list(fm_columns - keep)

        mask = mask_params & mask_r2
        df.loc[mask, cols_set_nan] = np.nan


def _return_fooof_values(df, column, freq_low=13, freq_high=35,
                         return_val='array'):
    """Filter FOOOF results (e.g. amplitudes, center freqs) for each freq band.

    Return array if return_max is False, otherwise return maximum value."""
    band_mask = ((df.fm_center_freqs >= freq_low)
                 & (df.fm_center_freqs <= freq_high))
    # Pro nan array: can perform array computations
    # while avoiding shape mismatch.
    # Pro nan float: can be used in df to filter
    nan_vals = np.nan  # choose nan float
    try:
        values = df[column][band_mask]
    except (TypeError, IndexError):
        values = nan_vals
    else:
        if len(values):
            if return_val == 'max':
                values = values.max()
            elif return_val == 'count':
                values = len(values)
            elif return_val == 'array':
                values = values
        elif not len(values):
            # only peaks outside of band range
            values = nan_vals
    return values


def _fooof_band_power(df, bands=None):
    """Get FOOOF results for each frequency band.

    Loop over all freq bands and original and normalized spectra and add
    the FOOOF arrays for center freqs, powers, and stds to df. Also add the
    maximum values for each band.

    Later add mean aperiodic power."""
    peaks_found = df.fm_center_freqs.notna()  # exclude FOOOF without peaks
    df_peaks = df[peaks_found]

    bands = cfg.BANDS.keys() if bands is None else bands
    for band in bands:
        if band == 'FTG':
            # added later
            continue
        (f_low, f_high) = cfg.BANDS[band]

        # Loop over getting arrays ("fm") and max values ("max")
        for return_value, descr in zip(['array', 'max'], ["", "_max"]):

            kwargs = dict(freq_low=f_low, freq_high=f_high, axis=1,
                          return_val=return_value)

            kwargs["column"] = "fm_center_freqs"
            col_cf = f"{band}_fm_centerfreqs{descr}"
            df[col_cf] = df_peaks.apply(_return_fooof_values, **kwargs)

            kwargs["column"] = "fm_standard_devs"
            col_std = f"{band}_fm_stds{descr}"
            df[col_std] = df_peaks.apply(_return_fooof_values, **kwargs)

            kwargs["column"] = "fm_powers"
            col_pwr = f"{band}_fm_powers{descr}"
            df[col_pwr] = df_peaks.apply(_return_fooof_values, **kwargs)

            kwargs["column"] = "fm_powers_log"
            col_pwr_log = f"{band}_fm_powers{descr}_log"
            df[col_pwr_log] = df_peaks.apply(_return_fooof_values, **kwargs)

            kwargs["column"] = "fm_gauss_powers"
            col_gauss = f"{band}_fm_gauss_powers{descr}"
            df[col_gauss] = df_peaks.apply(_return_fooof_values, **kwargs)

            kwargs["column"] = "fm_gauss_powers_log"
            col_gauss_log = f"{band}_fm_gauss_powers{descr}_log"
            df[col_gauss_log] = df_peaks.apply(_return_fooof_values, **kwargs)

            # Set peak powers to 0 if no peak was found
            rmv_empty = lambda x: (np.nan if not isinstance(x, float)
                                   and len(x) == 0 else x)
            df[col_cf] = df[col_cf].apply(rmv_empty)
            peak_fitted = df[col_cf].notna()
            no_peak_fit = df.fm_has_model & ~peak_fitted
            df.loc[no_peak_fit, col_pwr] = np.zeros(no_peak_fit.sum())
            df.loc[no_peak_fit, col_pwr_log] = np.zeros(no_peak_fit.sum())
            df.loc[no_peak_fit, col_std] = np.zeros(no_peak_fit.sum())
            df.loc[no_peak_fit, col_gauss] = np.zeros(no_peak_fit.sum())
            df.loc[no_peak_fit, col_gauss_log] = np.zeros(no_peak_fit.sum())

            # Check that number of extracted values is correct
            num_fitted_models = df.fm_has_model.sum()
            num_fitted_pwrs = df[col_pwr].notna().sum()
            num_fitted_pwrs_log = df[col_pwr_log].notna().sum()
            assert (num_fitted_models
                    == num_fitted_pwrs
                    == num_fitted_pwrs_log)
            num_model_band = df.loc[peak_fitted, 'fm_has_model'].sum()
            num_cfs = df[col_cf].notna().sum()
            assert num_model_band == num_cfs

        kwargs = dict(freq_low=f_low, freq_high=f_high, axis=1,
                      return_val='count')

        kwargs["column"] = "fm_center_freqs"
        col_cf = f"{band}_fm_peak_count"
        df[col_cf] = df_peaks.apply(_return_fooof_values, **kwargs)
        df[col_cf] = df[col_cf].fillna(0)  # replace nan with 0

        # Add mean aperiodic power and it's frequency
        centerfreqs = f"{band}_fm_centerfreqs_max"
        for row in df.dropna(subset="fm_exponent").itertuples():
            # add aperiodic mean power at band range
            aperiodic = row.fm_psd_ap_fit_log
            if not np.all(np.isfinite(aperiodic)):
                continue
            col_idx = f"{band}_fm_band_aperiodic_log"
            mask = (row.fm_freqs >= f_low) & (row.fm_freqs <= f_high)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ignore_warnings()
                df.loc[row.Index, col_idx] = aperiodic[mask].mean()

            # add aperiodic power at peak frequency
            cf_max = row.__getattribute__(centerfreqs)
            if np.isnan(cf_max):
                continue
            # band_max_idx = round(cf_max)
            band_max_idx = np.argmin(np.abs(row.fm_freqs - cf_max))
            col_idx = f"{band}_fm_fmax_aperiodic"
            df.loc[row.Index, col_idx] = row.fm_psd_ap_fit[band_max_idx]
    return df


def _add_FTGs(df):
    # combine gamma peaks with other fm_params (only one peak fitted maximum)
    df_gamma = df[df.fm_params == "gamma"]
    # add peak count
    peaks_found = df_gamma.fm_center_freqs.notna()
    df_peaks = df_gamma[peaks_found]
    f_low, f_high = df_gamma.fm_fit_range.dropna().values[0]
    kwargs = dict(freq_low=f_low, freq_high=f_high, axis=1,
                  return_val='count', column='fm_center_freqs')
    df_gamma["FTG_fm_peak_count"] = df_peaks.apply(_return_fooof_values,
                                                   **kwargs)
    df_gamma["FTG_fm_peak_count"] = df_gamma["FTG_fm_peak_count"].fillna(0)

    # Set peak powers to 0 if no peak was found
    ## TODO: For some reason this does not work..
    no_peak_fit = df_gamma.fm_has_model & ~peaks_found
    fill_zero = np.zeros(no_peak_fit.sum())
    df_gamma.loc[no_peak_fit, 'fm_powers'] = fill_zero
    df_gamma.loc[no_peak_fit, 'fm_powers_log'] = fill_zero
    df_gamma.loc[no_peak_fit, 'fm_gauss_powers'] = fill_zero
    df_gamma.loc[no_peak_fit, 'fm_gauss_powers_log'] = fill_zero
    df_gamma.loc[no_peak_fit, 'fm_standard_devs'] = fill_zero

    # make arrays FTG specific
    rename = {"fm_powers": "FTG_fm_powers_max",
              'fm_powers_log': 'FTG_fm_powers_max_log',
              'fm_gauss_powers': 'FTG_fm_gauss_powers_max',
              'fm_gauss_powers_log': 'FTG_fm_gauss_powers_max_log',
              'fm_center_freqs': 'FTG_fm_centerfreqs_max',
              'fm_standard_devs': 'FTG_fm_stds_max'}  # 'FTG_fm_stds_max'?
    df_gamma.rename(columns=rename, inplace=True)
    # make equalize arrays and maxima
    for col in rename.values():
        df_gamma[col] = df_gamma[col].astype(float)
    FTGs = [col for col in df_gamma if col.startswith('FTG')]
    comb = [df, df_gamma[FTGs]]
    df = pd.concat(comb, axis=1)
    return df


def auc(peak_power, standard_deviation):
    """Area under curve/integral for Gaussian function."""
    auc = np.sqrt(2) * np.sqrt(np.pi) * peak_power * standard_deviation
    return auc


def _area_under_curve_power(df, bands=None):
    """Add area under curve measure based on FOOOF results for each band."""
    bands = cfg.BANDS.keys() if bands is None else bands
    for band in bands:

        # Get powers
        pwr_gauss = f"{band}_fm_gauss_powers"
        pwr_peak = f"{band}_fm_powers"
        bandwidth_gauss = f"{band}_fm_stds"
        if band == 'FTG':
            pwr_gauss += '_max'
            pwr_peak += '_max'
            bandwidth_gauss += '_max'
        gauss_power = df[pwr_gauss]
        peak_power = df[pwr_peak]
        gauss_power_log = df[pwr_gauss + "_log"]
        peak_power_log = df[pwr_peak + "_log"]

        # Get bandwidth
        std_gauss = df[bandwidth_gauss]
        std_peak = df[bandwidth_gauss] * 2

        # AUC Gaussian
        col_gauss = f"{band}_fm_aucgauss"
        df[col_gauss] = auc(gauss_power, std_gauss)
        col_gauss_log = f"{band}_fm_aucgauss_log"
        df[col_gauss_log] = auc(gauss_power_log, std_gauss)

        # AUC Peak
        col_peak = f"{band}_fm_aucpeak"
        df[col_peak] = auc(peak_power, std_peak)
        col_peak_log = f"{band}_fm_aucpeak_log"
        df[col_peak_log] = auc(peak_power_log, std_peak)

        # AUC Gauss Sum
        calc_sum = lambda x: x.sum() if isinstance(x, np.ndarray) else x
        col_gauss_sum = f"{band}_fm_aucgauss_sum"
        df[col_gauss_sum] = df[col_gauss].apply(calc_sum)
        col_gauss_sum_log = f"{band}_fm_aucgauss_sum_log"
        df[col_gauss_sum_log] = df[col_gauss_log].apply(calc_sum)

        # AUC Peak Sum
        col_peak_sum = f"{band}_fm_aucpeak_sum"
        df[col_peak_sum] = df[col_peak].apply(calc_sum)
        col_peak_sum_log = f"{band}_fm_aucpeak_sum_log"
        df[col_peak_sum_log] = df[col_peak_log].apply(calc_sum)

        # AUC Gauss Max
        calc_max = lambda x: x.max() if isinstance(x, np.ndarray) else x
        col_gauss_max = f"{band}_fm_aucgauss_max"
        df[col_gauss_max] = df[col_gauss].apply(calc_max)
        col_gauss_max_log = f"{band}_fm_aucgauss_max_log"
        df[col_gauss_max_log] = df[col_gauss_log].apply(calc_max)

        # AUC Peak Max
        col_peak_max = f"{band}_fm_aucpeak_max"
        df[col_peak_max] = df[col_peak].apply(calc_max)
        col_peak_max_log = f"{band}_fm_aucpeak_max_log"
        df[col_peak_max_log] = df[col_peak_log].apply(calc_max)

        # Testing
        stds = f"{band}_fm_stds_max"
        assert df[col_gauss].notna().sum() == df[stds].notna().sum(), stds
    return df


def per_pwr_integral(df, func, f_low, f_high, log=False):
    col = 'fm_psd_peak_fit'
    log = '_log' if log else ''
    freqs = df.fm_freqs
    mask = (freqs >= f_low) & (freqs <= f_high)
    per_pwr = df[col + log]
    if np.isnan(per_pwr).all():
        return np.nan
    per_pwr = per_pwr[mask]
    return func(per_pwr)


def _periodic_integrated_power(df, bands=None):
    """Average periodic power for each band."""
    bands = cfg.BANDS.keys() if bands is None else bands
    for band in bands:
        f_low, f_high = cfg.BANDS[band]

        # Mean
        log = False
        func = np.mean
        col = f"{band}_fm_mean"
        args = (func, f_low, f_high, log)
        df.loc[:, col] = df.apply(per_pwr_integral, args=args, axis=1)
        log = True
        col_log = f"{band}_fm_mean_log"
        args = (func, f_low, f_high, log)
        df.loc[:, col_log] = df.apply(per_pwr_integral, args=args, axis=1)

        # Integral
        log = False
        func = np.sum
        col = f"{band}_fm_sum"
        args = (func, f_low, f_high, log)
        df.loc[:, col] = df.apply(per_pwr_integral, args=args, axis=1)
        log = True
        col_log = f"{band}_fm_sum_log"
        args = (func, f_low, f_high, log)
        df.loc[:, col_log] = df.apply(per_pwr_integral, args=args, axis=1)
    return df


def _signal_to_noise_ratio(df, bands=None):
    """Return SNR maximum.

    Important to calc SNR for all freqs, and then take maximum. Because
    beta_power_max=1.4 at 13 Hz might have lower SNR than beta_power_max=1.3
    at 30 Hz.

    Important to keep in mind: df.fm_psd_peak_fit is the fitted power spectrum
    and might very slightly deviate from the original power spectrum (df.psd).
    """
    df_broad = df[(df.fm_params == "broad") & df.fm_exponent.notna()]
    bands = cfg.BANDS.keys() if bands is None else bands
    for band in bands:
        (f_low, f_high) = cfg.BANDS[band]
        for row in df_broad.itertuples():

            snr = row.fm_psd_peak_fit / row.fm_psd_ap_fit
            mask = (row.fm_freqs >= f_low) & (row.fm_freqs <= f_high)

            if len(snr[mask]):
                snr_band_max = snr[mask].max()
            else:
                # all zeros
                snr_band_max = 0

            df.loc[row.Index, f"{band}_fm_SNR_max"] = snr_band_max

        assert (df[f"{band}_fm_SNR_max"].notna().sum()
                == df[(df.fm_params == "broad")].fm_psd_ap_fit.notna().sum()
                == df[(df.fm_params == "broad")].fm_psd_peak_fit.notna().sum()
                == df[(df.fm_params == "broad")].fm_exponent.notna().sum())


def _detect_noise_floor(df, show_plots=False, threshold=0.05):
    """
    Determine plateau onset and add to df.

    Important for reliable 1/f fitting."""
    # Fitting each psd in a while loop until convergence turned out to be
    # faster than applying a FOOOFGroup to all psds for all frequency ranges.
    fm = SpectralModel(max_n_peaks=1, min_peak_height=0.05,
                       peak_width_limits=(1, 2), verbose=False)
    # Frequency ranges: 5 Hz steps starting at 32 to avoid line noise freqs.
    # Plateau onsets above 100 Hz not relevant. (ECoG and EEG sometimes
    # plateaus above 600 Hz). 5 Hz stepsize determines plateau onset
    # resolution. 50 Hz frequency range seems sufficiently long while avoiding
    # several line noise peaks within one fit.
    frange_starts = list(range(32, 100, 5))
    frange_len = 50
    f_ranges = [(f_start, f_start+frange_len) for f_start in frange_starts]

    for row in df.dropna(subset="fm_exponent").itertuples():
        exponent = 1
        frange_idx = 0
        while exponent > threshold and frange_idx < len(f_ranges):

            psd_fit = row.psd
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ignore_warnings()
            if np.any(np.isnan(row.psd_log)) or np.any(np.isinf(row.psd_log)):
                # remove negative PSD values
                psd_fit[psd_fit <= 0] = 1e-12
            fm.fit(row.psd_freqs, row.psd, f_ranges[frange_idx])
            exponent = fm.get_params('aperiodic_params', 'exponent')
            frange_idx += 1
        plateau_onset = f_ranges[frange_idx-1][0]  # if exp < 0.05 else False
        df.loc[row.Index, "fm_plateau"] = plateau_onset
        if show_plots:
            print(f"{f_ranges[frange_idx-1]}: 1/f = {exponent:.2f}")
            fm.report(plot_range=[1, 100])


def _bispectral_index(df):
    """Calculate the bispectral index and at to df.

    Bispectral index = LFA_mean / HFA_mean.
    """
    lfa_mean = 1  # don't average, just take 1 Hz PSD value
    hfa_mean1 = [40, 45]  # avoid plateau, avoid line noise
    hfa_mean2 = [55, 60]  # avoid line noise, use maximum possible Gamma
    hfa_mean3 = [90, 95]  # high
    hfa_mean4 = [155, 160]  # plateau

    cols = ["psd_bispectral40", "psd_bispectral55",
            "psd_bispectral90", "psd_bispectral155"]
    HFAs = [hfa_mean1, hfa_mean2, hfa_mean3, hfa_mean4]

    freqs = df.psd_freqs.iloc[0]

    for col, hfa in zip(cols, HFAs):
        mask_lfa = (freqs == lfa_mean)
        mask_hfa = (freqs >= hfa[0]) & (freqs <= hfa[1])
        idx = lambda psd: psd[mask_lfa].mean() / psd[mask_hfa].mean()
        bispectral_idx = df.psd.apply(idx)
        df[col] = bispectral_idx


def _offon_change_kuhn(off_array, on_array):
    """Equation as in Kühn's publications. Also see Zaidel 2010 who explores
    this in depth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ignore_warnings()
        result = (off_array - on_array) / off_array * 100
    return result


def _offon_change_abs(off_array, on_array):
    """Calculate on-off difference in percentage."""
    return on_array - off_array


def _offon_change_abs(off_array, on_array):
    """Calculate on-off difference in percentage."""
    return off_array - on_array


def _offon_change_rel(off_array, on_array):
    """Calculate on-off difference in percentage.

    Complicated function a) to avoid divide by zero warnings and b) to avoid
    nan values in only one of both conditions.

    If off_array is 0, return 0."""
    # does not support arrays of arrays (PSD column)
    if isinstance(on_array[0], np.ndarray):
        return np.ones_like(off_array) * np.nan

    result = np.ones_like(off_array) * np.inf  # use inf to check later

    nan_mask = (np.isnan(off_array) | np.isnan(on_array))
    result[nan_mask] = np.nan

    # Divide by OFF if off not zero
    off_not_zero = (off_array != 0)
    divide_mask = ~nan_mask & off_not_zero
    diff = off_array[divide_mask] - on_array[divide_mask]
    result[divide_mask] = diff / off_array[divide_mask] * 100

    # Divide by ON if off zero and ON not zero to get negative improvement
    off_zero = (off_array == 0) & (on_array != 0)
    divide_mask = ~nan_mask & off_zero
    result[divide_mask] = -np.inf

    # Set improvement to zero if both are zero
    both_zero = (on_array == 0) & (off_array == 0)
    result[both_zero] = 0

    assert np.all(result[~np.isnan(result)] < np.inf), "You missed values!"
    return result


def _add_offon_difference(df):
    """Add UPDRS On-Off differences."""
    df_all = []
    for project in df.project.unique():
        df_proj = df[(df.project == project)]
        df_concat = _subtract_offon(df_proj)
        df_all.append(df_concat)
    df_concat = pd.concat(df_all).reset_index()
    return df_concat


def _subtract_offon(df):
    df_both, mask_off, mask_on = _prepare_df_offon(df)

    # add offon as a third condition instead of as extra columns
    df_offon_abs = df_both.loc[mask_off].copy()

    df_offon_abs.loc[:, 'cond'] = 'offon_abs'

    # Loop over all power and fm  and UPDRS columns
    cols = _get_cols_offon_difference(df)
    for col in cols:
        val_off = df_both.loc[mask_off, col].to_numpy()
        val_on = df_both.loc[mask_on, col].to_numpy()
        change_abs = _offon_change_abs(val_off, val_on)
        df_offon_abs.loc[:, col] = change_abs

    # Add relative UPDRS improvement to df and subtracted LFP spectra
    cols = _get_cols_offon_difference(df, UPDRS_only=True)
    df_offon_rel = df_offon_abs.copy()
    df_offon_rel.loc[:, 'cond'] = 'offon_rel'
    for col in cols:
        val_off = df_both.loc[mask_off, col].to_numpy()
        val_on = df_both.loc[mask_on, col].to_numpy()

        # change_rel = _offon_change_rel(val_off, val_on)
        change_rel = _offon_change_kuhn(val_off, val_on)
        df_offon_rel.loc[:, col] = change_rel

    # set offon chs as bad if bad in ON or OFF
    group_cols = ["subject", "ch_nme"]
    group = df.groupby(group_cols)
    group_any_bad = group.ch_bad.apply(any)
    df_offon_abs = df_offon_abs.set_index(group_cols)
    df_offon_abs.loc[group_any_bad, "ch_bad"] = True
    df_offon_abs = df_offon_abs.reset_index()

    df_offon_rel = df_offon_rel.set_index(group_cols)
    df_offon_rel.loc[group_any_bad, "ch_bad"] = True
    df_offon_rel = df_offon_rel.reset_index()

    df_concat = pd.concat([df, df_offon_abs, df_offon_rel])
    df_concat = df_concat.reset_index(drop=True)

    return df_concat


def _get_cols_offon_difference(df, UPDRS_only=False):
    """Return list of columns for which on-off diff should be calculated."""
    updrs_cols = [col for col in df.columns if col.startswith("UPDRS_")]
    fm_cols = ["fm_exponent", "fm_offset", 'fm_offset_log', "fm_knee",
               "fm_knee_fit", 'fm_knee_log']
    psd_cols = [col for col in df.columns if 'psd' in col]
    psd_cols += ['asd', 'fm_fooofed_spectrum', 'fm_fooofed_spectrum_log']
    psd_cols.remove('psd_freqs')
    psd_cols.remove('psd_method')
    psd_cols.remove('psd_kind')
    if UPDRS_only:
        cols = updrs_cols
    else:
        cols = updrs_cols + fm_cols + psd_cols
        for band in list(cfg.BANDS.keys()):
            cols.extend([col for col in df.columns if col.startswith(band)
                         # drop arrays and strings
                         and df[col].dtypes == float])
    # drop columns which don't exist
    cols = [col for col in cols if col in df.columns]
    return cols


def _prepare_df_offon(df):
    """Prepare dataframe for on-off difference calculation."""
    # ignore monopolar reference and bad channels
    mask = ((df.ch_reference != "monopolar") & (df.bids_task != 'noise'))
    df_both = df[mask].sort_values(["subject", "ch_nme", "cond"])
    # SORT VALUES INCREDIBLY IMPORTANT

    # only consider channels that are equal in both conditions for each
    # subject.
    # bids_acquisition crucial for Tan because some sessions only in one cond,
    # however, does not work with Neumann data.
    group_cols = ["subject", "ch_nme"]
    group = df_both.groupby(group_cols)
    group_both_conds = (group.cond.unique().apply(len) == 2)
    group_both = group_both_conds  # keep bad channels
    df_both = df_both.set_index(group_cols)
    df_both = df_both.loc[group_both]
    df_both = df_both.reset_index()

    mask_off = (df_both.cond == "off")
    mask_on = (df_both.cond == "on")
    assert mask_on.sum() == mask_off.sum()
    assert mask_on.shape == mask_off.shape
    assert np.all(mask_on != mask_off)

    return df_both, mask_off, mask_on


def _drop_redundant_columns(df):
    """Drop uninformative columns that contain the same value for all rows."""
    df_sub = df[df.subject != "NeuEmptyroom"]
    for col in df.columns:
        if col == "subject":
            continue
        try:
            length = len(df_sub[col].unique())
        except TypeError:
            continue
        if length == 1:
            # print(col)
            # print(df_sub[col].unique()[0])
            # print("="*30)
            df.drop(columns=col, inplace=True)


def _organize_df(df):
    df.reset_index(inplace=True, drop=True)

    # Concatenate all fm_params to single df #################################
    no_duplicates = ["subject", "ch_nme", "cond", "fm_params",
                     "psd_kind", "psd_method", "ch_reference", 'bids_task',
                     'bids_acquisition', 'bids_processing']

    # Merge HFO columns with broad
    HFO_cols = [col for col in df.columns if col.startswith("HFO")
                and 'fm' in col]
    df_HFO = df.loc[(df.fm_params == "HFO"), no_duplicates + HFO_cols]
    df_HFO['fm_params'] = "broad"
    df = df[df.fm_params != "HFO"]
    df = df.drop(columns=HFO_cols)
    df_all = df.merge(df_HFO, on=no_duplicates, how='outer')

    # Merge fm_exponent narrow with broad and lorentzian
    keep_cols = no_duplicates + ['fm_exponent']
    df_narrow = df.loc[(df.fm_params == "narrow"), keep_cols]
    rename = {"fm_exponent": "fm_exponent_narrow"}
    df_narrow.rename(columns=rename, inplace=True)
    df_narrow['fm_params'] = "broad"
    df_narrow2 = df_narrow.copy()
    df_narrow2['fm_params'] = "lor100"
    df_narrow = pd.concat([df_narrow, df_narrow2], axis=0)
    df_all = df_all[df_all.fm_params != "narrow"]
    df_all = df_all.merge(df_narrow, on=no_duplicates, how='outer')

    # Merge gamma with broad
    FTG_cols = [col for col in df.columns if col.startswith("FTG")]
    df_FTG = df.loc[(df.fm_params == "gamma"), no_duplicates + FTG_cols]
    df_FTG['fm_params'] = "broad"
    df_all = df_all[df_all.fm_params != "gamma"]
    df_all = df_all.drop(columns=FTG_cols)
    df_all = df_all.merge(df_FTG, on=no_duplicates, how='outer')
    df = df_all

    # sort columns alphabetically
    df = df.reindex(sorted(df.columns, key=str.lower), axis=1)

    # put a few specific columns to the front
    cols_first = ["subject", "ch_nme", "cond", "title"]
    df = df[cols_first + [col for col in df.columns if col not in cols_first]]
    df = df.sort_values(cols_first)

    # make sure there are no duplicate rows
    cols = ['subject', "ch_nme", "title", "fm_params", "psd_kind",
            "psd_method", "ch_reference", "cond", 'bids_task',
            'bids_acquisition', 'bids_processing', 'project']
    msg = "Duplicates found!"
    assert len(df[df.duplicated(subset=cols)]) == 0, msg
    return df  # cannot change order in_place, therefore must return df


if __name__ == "__main__":
    # measure time to execute this function:
    import time
    start = time.time()
    organize_df()
    end = time.time()
    print(f"Time elapsed: {end - start:.2f} s")