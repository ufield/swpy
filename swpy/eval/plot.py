import matplotlib.pyplot as plt

def create_plt_gp_result(result_dst_df, result_omni_df, event_dt):
    fig, ax = plt.subplots(4, 1, figsize=(14,12))

    plt.rcParams["font.size"] = 18
    ax[0].set_title('Storm Event (' + event_dt + ')')

    # plot Dst --------
    result_dst_df.plot('Time', y=['DST[nT]', 'DST_pred'], color=['black', 'red'], ax=ax[0])
    ax[0].fill_between(result_dst_df['Time'].values, result_dst_df['cr_lower'], result_dst_df['cr_upper'], alpha=0.5)

    ax[0].set_ylabel('DST[nT]')
    ax[0].legend(['Observed', '4 hours ahead prediction', 'Confidence'])
    ax[0].xaxis.set_visible(False)

    # plot Bz & |B| ---
    result_omni_df.plot('Time', y=['F', 'BZ_GSM'], color=['black', 'red'], ax=ax[1])
    ax[1].legend(['|B|', 'Bz_GSM'])
    ax[1].set_ylabel('nT')
    ax[1].xaxis.set_visible(False)

    # plot Flow speed
    result_omni_df.plot('Time', y=['flow_speed'], color='black', ax=ax[2])
    ax[2].get_legend().remove()
    ax[2].set_ylabel('V [km/s]')
    ax[2].xaxis.set_visible(False)

    # plot proton_density
    result_omni_df.plot('Time', y=['proton_density'], color='black', ax=ax[3])
    ax[3].get_legend().remove()
    ax[3].set_ylabel('N [cm^-3]')
    ax[3].xaxis.label.set_visible(False)

    return plt