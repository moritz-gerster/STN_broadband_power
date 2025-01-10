from scripts.plot_figures._sweetspot_distance import plot_sweetspot_distance


def figure5(df_orig):
    df_corr = plot_sweetspot_distance(df_orig, fig_dir='Figure5')
    return df_corr