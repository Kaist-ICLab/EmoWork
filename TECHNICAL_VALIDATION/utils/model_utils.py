from biosppy.signals import eda, eeg
from biosppy.signals import tools as st
from biosppy import utils, plotting
from scipy import signal
import pandas as pd
import numpy as np
import scipy.stats as stats
import utils.cvxEDA as cvxEDA
import warnings
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from cvxopt import solvers
from scipy.stats import linregress

warnings.simplefilter(action='ignore', category=FutureWarning)


"""
EDA data
"""
def eda_mean(df, window, slide):
    """
    (1) Filtering: first-order Butterworth low-pass filter
    (2) Compute mean of Phasic and Tonic component in each window  
    Arguments
    ---------
        df: Dataframe
            EDA signal and timestamp data
        window: int
            window size (ms)
        slide: int
            slide size (ms)

    Returns
    -------
        eda_mean: Dataframe
            timestamp, phasic_mean, tonic_mean
    """

    rawEDA = df['eda']
    timestamp = df['Timestamp']

    # Filtering: first-order Butterworth low-pass filter

    order = 1
    cut_off_freq = 0.4 
    # 오리지널 코드임 # b,a = signal.butter(order, cut_off_freq, 'low', analog=True) # low pass filter
    '''두리 임시로추가 '''
    fs = 4
    b, a = signal.butter(order, cut_off_freq, btype='low', fs=fs) 
    ''' '''

    lp_filtered = signal.filtfilt(b,a, rawEDA).tolist()

    r, p, t, l, d, e, obj  = cvxEDA.cvxEDA(lp_filtered, 0.25, options={'show_progress':False})

    """
        r: phasic component
        p: sparse SMNA driver of phasic component
        t: tonic component
        l: coefficients of tonic spline
        d: offset and slope of the linear drift term
        e: model residuals
        obj: value of objective function being minimized (eq 15 of paper)
    """

    ### Binning mean values
    eda_scr = pd.DataFrame(data={'Timestamp': timestamp})
    eda_scr['eda'] = df['eda']
    eda_scr['phasic'] = r
    eda_scr['tonic'] = t

    new_phasic_means = []
    new_tonic_means = [] 
    new_timestamp = [] 


    ### Binning based on the timestamp (using only timestamp data)

    df_out = pd.DataFrame(columns=['Timestamp',
                                   'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 
                                   'phasic_mean', 'phasic_std', 'phasic_min', 'phasic_max', 
                                   'tonic_mean', 'tonic_std', 'tonic_min', 'tonic_max'])
    window = window * 1000
    slide = slide * 1000

    window_start = eda_scr['Timestamp'].min()

    while True:
        window_end  = window_start + window
        window_df = eda_scr.loc[(eda_scr['Timestamp'] > window_start) & (eda_scr['Timestamp'] < window_end)]

        temp_dict = dict()
        temp_dict['Timestamp'] = window_start

        try:
            temp_dict['EDA_mean'] = window_df['eda'].mean()
            temp_dict['EDA_std'] = window_df['eda'].std()
            temp_dict['EDA_min'] = window_df['eda'].min()
            temp_dict['EDA_max'] = window_df['eda'].max()

            temp_dict['phasic_mean'] = window_df['phasic'].mean()
            temp_dict['phasic_min'] = window_df['phasic'].min()
            temp_dict['phasic_max'] = window_df['phasic'].max()
            temp_dict['phasic_std'] = window_df['phasic'].std()

            temp_dict['tonic_mean'] = window_df['tonic'].mean()
            temp_dict['tonic_min'] = window_df['tonic'].min()
            temp_dict['tonic_max'] = window_df['tonic'].max()
            temp_dict['tonic_std'] = window_df['tonic'].std()

            time_values = np.arange(len(window_df))
            eda_raw_data = np.array(window_df['eda'])
            phasic_raw_data = np.array(window_df['phasic'])
            tonic_raw_data = np.array(window_df['tonic'])

            eda_slope, intercept, r_value, p_value, std_err = linregress(time_values, eda_raw_data)
            phasic_slope, intercept, r_value, p_value, std_err = linregress(time_values, phasic_raw_data)
            tonic_slope, intercept, r_value, p_value, std_err = linregress(time_values, tonic_raw_data)

            temp_dict['EDA_slope'] = eda_slope.mean()
            # temp_dict['phasic_slope'] = phasic_slope.mean()
            # temp_dict['tonic_slope'] = tonic_slope.mean()

        except:
            for col in df_out.columns.drop('Timestamp'):
                temp_dict[col] = np.nan

        df_out.loc[len(df_out)] = temp_dict
        window_start += slide
        if (window_start + window) > eda_scr['Timestamp'].max():
            break

    df_out.reset_index(drop=True, inplace=True)

    # Normalization
    for col in df_out.columns.drop('Timestamp'):
        scaler = MinMaxScaler()
        scaler.fit(df_out[col].values.reshape(-1,1))
        df_out[col] = scaler.transform(df_out[col].values.reshape(-1,1))
    # normEDA = stats.zscore(lp_filtered)
        
    return df_out


def eda_count_peaks(df, window, slide):
    """

    (1) Filtering: first-order Butterworth low-pass filter
    (2) Counting peaks: KBK_SCR 
        - Differetiating (np.diff)
        - Smoothing (Barlett)
        - Fining consecutive zero-crossings
    (3) Binning

    Arguments
    ---------
        df: Dataframe
            EDA signal and timestamp data
        window: int
            window size (ms)
        slide: int
            slide size (ms)

    Returns
    ------- 
        eda_peaks: Dataframe
            timestamp, peak_per_window
    """

    rawEDA = df['eda']
    timestamp = df['Timestamp']

    # Filtering: first-order Butterworth low-pass filter
    order = 1
    cut_off_freq = 0.4 
    b,a = signal.butter(order, cut_off_freq, 'low', analog=True) # low pass filter
    lp_filtered = signal.filtfilt(b,a, rawEDA).tolist()

    # Counting peaks: KBK_SCR (https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy.signals.eda.kbk_scr)
    out = eda.kbk_scr(lp_filtered, sampling_rate = 4, min_amplitude = 0.1) # onsets, peaks, amplitudes
    peaks = out['peaks'].tolist()

    # Binning peaks
    eda_scr = pd.DataFrame(data={'Timestamp': timestamp})
    eda_scr = eda_scr.reset_index(drop=True)

    new_col = np.zeros(len(rawEDA))
    new_col[peaks] = 1
    eda_scr['peak'] = new_col

    new_peaks = []
    new_timestamp = []

    ### Binning based on the timestamp (using only timestamp data)

    df_out = pd.DataFrame(columns=['Timestamp','peak'])
    window = window * 1000
    slide = slide * 1000

    window_start = eda_scr['Timestamp'].min()

    while True:
        window_end  = window_start + window
        window_df = eda_scr.loc[(eda_scr['Timestamp'] > window_start) & (eda_scr['Timestamp'] < window_end)]

        temp_dict = dict()
        temp_dict['Timestamp'] = window_start

        try:
            temp_dict['peak'] = window_df['peak'].mean()
        except:
            temp_dict['peak'] = np.nan

        df_out.loc[len(df_out)] = temp_dict
        window_start += slide
        if (window_start + window) > eda_scr['Timestamp'].max():
            break

    df_out.reset_index(drop=True, inplace=True)

    for col in df_out.columns.drop('Timestamp'):
        scaler = MinMaxScaler()
        scaler.fit(df_out[col].values.reshape(-1,1))
        df_out[col] = scaler.transform(df_out[col].values.reshape(-1,1))
        
    return df_out



"""
EEG data
"""

def eeg_mean_bandpower(df, sampling_rate, window, slide, labels=None, path=None, show=True): 
    #https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#gamb08
    """
    (1) Filtering
        - High pass filtering: Butterworth high-pass filter (order=8, frequency=4)
        - Low pass filtering: Butterworth low_pass filter (order=16, frequency=40)

    (2) Extract Band power features: compute band power features for each window 
    (Average power in each frequency band: theta (4-8Hz), alpha_low (8-10Hz), alpha_high(10-13Hz), beta(13-25Hz), gamma(25-40Hz))
        - Fourier Transform → Compute the power spectrum of a signal 
        - # of sensors: 4, # of frequency band: 5 ⇒ 4*5 = 20 signals

    Arguments
    ----------
        signal : array
            Raw EEG signal matrix; each column is one EEG channel.
        sampling_rate : int, float, optional
            Sampling frequency (Hz).
        labels : list, optional
            Channel labels.
        path : str, optional
            If provided, the plot will be saved to the specified file.
        show : bool, optional
            If True, show a summary plot.
    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered EEG signal.
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    plf_pairs : list
        PLF pair indices.
    plf : array
        PLF matrix; each column is a channel pair.
    """
    
    data_extracting_window = 0.1
    data_extracting_overlap = 0    

    signal = df[['TP9', 'AF7', 'AF8', 'TP10']]
    timestamp = df['Timestamp'].tolist()

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    signal = np.reshape(signal, (signal.shape[0], -1))

    sampling_rate = float(sampling_rate)
    nch = signal.shape[1]

    if labels is None:
        labels = ["Ch. %d" % i for i in range(nch)]
    else:
        if len(labels) != nch:
            raise ValueError(
                "Number of channels mismatch between signal matrix and labels."
            )

    # high pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="highpass",
        order=8,
        frequency=4,
        sampling_rate=sampling_rate,
    )

    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

    # low pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="lowpass",
        order=16,
        frequency=40,
        sampling_rate=sampling_rate,
    )

    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)

    # band power features
    out = get_power_features(
        signal=filtered, sampling_rate=sampling_rate, window=data_extracting_window, overlap=data_extracting_overlap
    )

    ts_feat = out["ts"]
    theta = out["theta"]
    alpha_low = out["alpha_low"]
    alpha_high = out["alpha_high"]
    beta = out["beta"]
    gamma = out["gamma"]

    # If the input EEG is single channel do not extract plf
    # Initialises plf related vars for input and output requirement of plot_eeg function in case of nch <=1
    plf_pairs = []
    plf = []
    if nch > 1:
        # PLF features
        _, plf_pairs, plf = eeg.get_plf_features(
            signal=filtered, sampling_rate=sampling_rate, size=window, overlap=0.5
        )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_eeg(
            ts=ts,
            raw=signal,
            filtered=filtered,
            labels=labels,
            features_ts=ts_feat,
            theta=theta,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            beta=beta,
            gamma=gamma,
            plf_pairs=plf_pairs,
            plf=plf,
            path=path,
            show=True,
        )

    ### Binning based on the timestamp
    eeg_bp = pd.DataFrame()
    eeg_bp['Timestamp'] = timestamp[0]+(out['ts']*1000)
    eeg_bp[['theta1', 'theta2', 'theta3', 'theta4']] = out['theta']
    eeg_bp[['alpha_low1', 'alpha_low2', 'alpha_low3', 'alpha_low4']] = out["alpha_low"]
    eeg_bp[['alpha_high1', 'alpha_high2', 'alpha_high3', 'alpha_high4']] = out["alpha_high"]
    eeg_bp[['beta1', 'beta2', 'beta3', 'beta4']] = out["beta"]
    eeg_bp[['gamma1', 'gamma2', 'gamma3', 'gamma4']] = out["gamma"]

    ### Binning based on the timestamp (using only timestamp data)

    eeg_cols = eeg_bp.columns.tolist()

    df_out = pd.DataFrame(columns=eeg_cols)
    window = window * 1000
    slide = slide * 1000

    window_start = eeg_bp['Timestamp'].min()

    while True:
        window_end  = window_start + window
        window_df = eeg_bp.loc[(eeg_bp['Timestamp'] > window_start) & (eeg_bp['Timestamp'] < window_end)]

        temp_dict = dict()
        temp_dict['Timestamp'] = window_start

        columns = window_df.columns.drop('Timestamp')
        for column in columns:
            temp_dict[column] = window_df[column].mean()
    
        df_out.loc[len(df_out)] = temp_dict

        window_start += slide
        if (window_start + window) > eeg_bp['Timestamp'].max():
            break
    df_out.reset_index(drop=True, inplace=True)
    
    # output
    args = (
        ts,
        filtered,
        ts_feat,
        theta,
        alpha_low,
        alpha_high,
        beta,
        gamma,
        plf_pairs,
        plf,
    )
    names = (
        "ts",
        "filtered",
        "features_ts",
        "theta",
        "alpha_low",
        "alpha_high",
        "beta",
        "gamma",
        "plf_pairs",
        "plf",
    )


    # return utils.ReturnTuple(args, names)

     # Normalization
    norm_eeg_bp = df_out.apply(stats.zscore)
    norm_eeg_bp['Timestamp'] = df_out['Timestamp']

    return norm_eeg_bp



"""
Temperature
"""

def temp_mean(df, window, slide):
    temp_data = df

    # Normalization
    # temp_data['temp'] = stats.zscore(temp_data[' temp'])

    df_out = pd.DataFrame(columns=['Timestamp','temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_slope'])
    window = window * 1000
    slide = slide * 1000
    window_start = temp_data['Timestamp'].min()

    while True:
        window_end = window_start + window
        window_df = temp_data.loc[(temp_data['Timestamp'] > window_start) & (temp_data['Timestamp'] < window_end)]

        row = []
        for col in ['temp']:
            mean_value = window_df[col].mean()
            std_value = window_df[col].std()
            min_value = window_df[col].min()
            max_value = window_df[col].max()

            time_values = np.arange(len(window_df))
            temperature_raw_data = np.array(window_df['temp'])
            slope, intercept, r_value, p_value, std_err = linregress(time_values, temperature_raw_data)
            

            row.extend([mean_value, std_value, min_value, max_value, slope.mean()])
        
        row.insert(0, window_start)
        df_out.loc[len(df_out)] = row

        window_start += slide
        if (window_start + window) > temp_data['Timestamp'].max():
            break

    for col in df_out.columns.drop('Timestamp'):
        df_out[col] = stats.zscore(df_out[col])

    df_out.reset_index(drop=True, inplace=True)

    return df_out


"""
Accelerometer
"""
def acc_mag_mean(df, window, slide):
    acc_data = df 

    magnitude = np.sqrt(acc_data['accX']**2 + acc_data['accY']**2 + acc_data['accZ']**2)
    acc_data['magnitude'] = magnitude

    # Normalization
    acc_data['magnitude'] = stats.zscore(acc_data['magnitude'])

    df_out = pd.DataFrame(columns=['Timestamp',
                                   'accX_mean', 'accX_std',
                                   'accY_mean', 'accY_std',
                                   'accZ_mean', 'accZ_std', 
                                    'acc_magnitude'])
    window = window * 1000
    slide = slide * 1000
    window_start = acc_data['Timestamp'].min()

    while True:
        window_end = window_start + window
        window_df = acc_data.loc[(acc_data['Timestamp'] > window_start) & (acc_data['Timestamp'] < window_end)]

        row = []
        # mean_value = window_df[col].mean()

        accX_mean = window_df['accX'].mean()
        accX_std = window_df['accX'].std()
        accY_mean = window_df['accY'].mean()
        accY_std = window_df['accY'].std()
        accZ_mean = window_df['accZ'].mean()
        accZ_std = window_df['accZ'].std()
        mag_mean = window_df['magnitude'].mean()

        row.extend([accX_mean, accX_std, accY_mean, accY_std, accZ_mean, accZ_std, mag_mean])
        
        row.insert(0, window_start)
        df_out.loc[len(df_out)] = row

        window_start += slide
        if (window_start + window) > acc_data['Timestamp'].max():
            break

    for col in df_out.columns.drop('Timestamp'):
        df_out[col] = stats.zscore(df_out[col])

    # df_out.add_prefix('mean_')
    df_out.reset_index(drop=True, inplace=True)

    return df_out


"""
ECG RR data
"""

def ecg_hrv(df, window_size, slide):
    window_size = window_size*1000
    slide = slide*1000

    df_out = pd.DataFrame(columns=['Timestamp', 'ibi','bpm','sdnn','rmssd','pnn50'])
    window_start = df['Timestamp'].min() 

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] > window_start) & (df['Timestamp'] < window_end)]
        rr_list = window_df['RR-interval']
        rr_diff = np.abs(np.diff(rr_list))
        rr_sqdiff = np.power(rr_diff, 2)
        wd, m = calc_ts_measures(rr_list, rr_diff, rr_sqdiff)
        m['Timestamp'] = window_start
        df_out.loc[len(df_out)] = m
        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break
    old_ts = df_out['Timestamp'].tolist()
    # new_ts =[int(x) if x.is_integer() else x for x in old_ts]
    new_ts = [int(x) if isinstance(x, float) and x.is_integer() else x for x in old_ts]
    new_dt = [datetime.fromtimestamp(x/1000) for x in new_ts]
    df_out['Datetime'] = new_dt

    # Normalization
    df_out['ibi'] = stats.zscore(df_out['ibi'])
    df_out['bpm'] = stats.zscore(df_out['bpm'])
    df_out['sdnn'] = stats.zscore(df_out['sdnn'])
    if np.isnan(stats.zscore(df_out['rmssd'])).any():
        df_out['pnn50'] = 0
    else:
        df_out['rmssd'] = stats.zscore(df_out['rmssd'])
    if np.isnan(stats.zscore(df_out['pnn50'])).any():
        df_out['pnn50'] = 0
    else:
        df_out['pnn50'] = stats.zscore(df_out['pnn50'])

    df_out.reset_index(drop=True, inplace=True)

    return df_out


"""
ECG HR data
"""
def ecg_hr(df, window, slide):
    window_size = window*1000
    slide = slide*1000

    df_out = pd.DataFrame(columns=['Timestamp','hr_mean', 'hr_std'])
    window_start = df['Timestamp'].min()

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] > window_start) & (df['Timestamp'] < window_end)]
        m = dict()
        m['Timestamp'] = window_start
        try:
            m['hr_mean'] = window_df['HR'].mean()
            m['hr_std'] = window_df['HR'].std()
        except:
            m['hr_mean'] = np.nan
            m['hr_std'] = np.nan
        df_out.loc[len(df_out)] = m
        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break

    old_ts = df_out['Timestamp'].tolist()
    # new_ts =[int(x) if x.is_integer() else x for x in old_ts]
    new_ts = [int(x) if isinstance(x, float) and x.is_integer() else x for x in old_ts]
    new_dt = [datetime.fromtimestamp(x/1000) for x in new_ts]
    df_out['Datetime'] = new_dt

    # Normalization
    df_out['hr_mean'] = stats.zscore(df_out['hr_mean'])
    df_out['hr_std'] = stats.zscore(df_out['hr_std'])

    df_out.reset_index(drop=True, inplace=True)

    return df_out        


"""
Audio data
"""
def mean_audio(df, window_size, slide):
    window_size = window_size * 1000
    slide = slide * 1000

    #print(df.head(2))

    cols = [col + '_' + stat for col in df.columns[:-1] for stat in ['mean']]
    cols.insert(0, 'Timestamp')
    mean_out = pd.DataFrame(columns=cols)

    window_start = df['Timestamp'].min()

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)]

        row = []
        for column in window_df.columns[:-1]:  
            mean_value = window_df[column].mean()
            row.extend([mean_value])

        row.insert(0, window_start)
        mean_out.loc[len(mean_out)] = row

        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break

    # Normalization
    mean_out_time = mean_out['Timestamp']


    scaler = StandardScaler()
    scaled_mean_out = pd.DataFrame(scaler.fit_transform(mean_out), columns=mean_out.columns)
    scaled_mean_out['Timestamp'] = mean_out_time
    
    scaled_mean_out.reset_index(drop=True, inplace=True)

    #print(scaled_mean_out.head(2))

    return scaled_mean_out

def min_audio(df, window_size, slide):
    window_size = window_size * 1000
    slide = slide * 1000

    cols = [col + '_' + stat for col in df.columns[:-1] for stat in ['min']]
    cols.insert(0, 'Timestamp')
    min_out = pd.DataFrame(columns=cols)

    window_start = df['Timestamp'].min()

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)]

        row = []
        for column in window_df.columns[:-1]:  # 마지막 'Timestamp' 컬럼을 제외합니다.
            min_value = window_df[column].min()
            row.extend([min_value])

        row.insert(0, window_start)
        min_out.loc[len(min_out)] = row

        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break

    # Normalization
    min_out_time = min_out['Timestamp']

    scaler = StandardScaler()
    scaled_min_out = pd.DataFrame(scaler.fit_transform(min_out), columns=min_out.columns)
    scaled_min_out['Timestamp'] = min_out_time
    
    scaled_min_out.reset_index(drop=True, inplace=True)

    return scaled_min_out

def max_audio(df, window_size, slide):
    window_size = window_size * 1000
    slide = slide * 1000

    cols = [col + '_' + stat for col in df.columns[:-1] for stat in ['max']]
    cols.insert(0, 'Timestamp')
    max_out = pd.DataFrame(columns=cols)

    window_start = df['Timestamp'].min()

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)]

        row = []
        for column in window_df.columns[:-1]:  # 마지막 'time' 컬럼을 제외합니다.
            max_value = window_df[column].max()
            row.extend([max_value])

        row.insert(0, window_start)
        max_out.loc[len(max_out)] = row

        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break

    max_out.reset_index(drop=True, inplace=True)

    # Normalization
    max_out_time = max_out['Timestamp']

    scaler = StandardScaler()
    scaled_max_out = pd.DataFrame(scaler.fit_transform(max_out), columns=max_out.columns)
    scaled_max_out['Timestamp'] = max_out_time
    
    scaled_max_out.reset_index(drop=True, inplace=True)

    return scaled_max_out

def std_audio(df, window_size, slide):
    window_size = window_size * 1000
    slide = slide * 1000
    cols = [col + '_' + stat for col in df.columns[:-1] for stat in ['std']]
    cols.insert(0, 'Timestamp')
    std_out = pd.DataFrame(columns=cols)
    window_start = df['Timestamp'].min()

    while True:
        window_end = window_start + window_size
        window_df = df.loc[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)]

        row = []
        for column in window_df.columns[:-1]:  # 마지막 'time' 컬럼을 제외합니다.
            std_value = window_df[column].std()
            row.extend([std_value])

        row.insert(0, window_start)
        std_out.loc[len(std_out)] = row

        window_start += slide
        if (window_start + window_size) > df['Timestamp'].max():
            break

    # Normalization
    std_out_time = std_out['Timestamp']
    scaler = StandardScaler()
    scaled_std_out = pd.DataFrame(scaler.fit_transform(std_out), columns=std_out.columns)
    scaled_std_out['Timestamp'] = std_out_time
    
    scaled_std_out.reset_index(drop=True, inplace=True)

    return scaled_std_out



"""
Helper
"""

def label_bin(df, label, window, slide):
    eda_scr = df

    df_out = pd.DataFrame(columns=['Timestamp', label])
    window = window * 1000
    slide = slide * 1000

    window_start = eda_scr['Timestamp'].min()
    while True:
        window_end  = window_start + window
        window_df = eda_scr.loc[(eda_scr['Timestamp'] > window_start) & (eda_scr['Timestamp'] < window_end)]
        
        row = []
        mean_value = window_df[label].mean()
        row.extend([mean_value])
        row.insert(0, window_start)
        df_out.loc[len(df_out)] = row

        window_start += slide
        if (window_start + window) > eda_scr['Timestamp'].max():
            break

    df_out.reset_index(drop=True, inplace=True)

    return df_out



def signal_interp(df, sampling_rate):
    raw_t = df['Timestamp'].tolist()
    x = np.arange(raw_t[0], raw_t[-1], 1000/sampling_rate)
    y = df

    x_df = pd.DataFrame(x, columns = ['Timestamp'])
    temp_df = pd.DataFrame()
    temp_df['Timestamp']= x_df

    result = pd.merge_asof(temp_df, y, on='Timestamp', direction='nearest')

    old_ts = result['Timestamp'].tolist()
    new_ts =[int(x) if x.is_integer() else x for x in old_ts]
    new_dt = [datetime.fromtimestamp(x/1000) for x in new_ts]
    result['Datetime'] = new_dt


    result = result.interpolate(method='pad')

    return result 

    

def get_power_features(signal=None, sampling_rate=1000.0, window=1, overlap=0):
    #https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#gamb08

    """Extract band power features from EEG signals.
    Computes the average signal power, with overlapping windows, in typical
    EEG frequency bands:
    * Theta: from 4 to 8 Hz,
    * Lower Alpha: from 8 to 10 Hz,
    * Higher Alpha: from 10 to 13 Hz,
    * Beta: from 13 to 25 Hz,
    * Gamma: from 25 to 40 Hz.
    Parameters
    ----------
    signal  array
        Filtered EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    window : float, optional
        Window size (seconds).
    slide : float, optional
        Slide size (seconds).
    Returns
    -------
    ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    nch = signal.shape[1]

    sampling_rate = float(sampling_rate)

    # convert sizes to samples
    window = int(window * sampling_rate)
    step = window - int(overlap * window)


    # padding
    min_pad = 1024
    pad = None
    if window < min_pad:
        pad = min_pad - window

    # frequency bands
    bands = [[4, 8], [8, 10], [10, 13], [13, 25], [25, 40]]
    nb = len(bands)

    # windower
    fcn_kwargs = {"sampling_rate": sampling_rate, "bands": bands, "pad": pad}
    index, values = st.windower(
        signal=signal,
        size=window,
        step=step,
        kernel="hann",
        fcn=eeg._power_features,
        fcn_kwargs=fcn_kwargs,
    )

    # median filter
    md_size = int(0.625 * sampling_rate / float(step))
    if md_size % 2 == 0:
        # must be odd
        md_size += 1

    for i in range(nb):
        for j in range(nch):
            values[:, i, j], _ = st.smoother(
                signal=values[:, i, j], kernel="median", size=md_size
            )

    # extract individual bands
    theta = values[:, 0, :]
    alpha_low = values[:, 1, :]
    alpha_high = values[:, 2, :]
    beta = values[:, 3, :]
    gamma = values[:, 4, :]

    # convert indices to seconds
    ts = index.astype("float") / sampling_rate

    # output
    args = (ts, theta, alpha_low, alpha_high, beta, gamma)
    names = ("ts", "theta", "alpha_low", "alpha_high", "beta", "gamma")

    return utils.ReturnTuple(args, names)

def calc_ts_measures(rr_list, rr_diff, rr_sqdiff, measures={}, working_data={}):
    '''calculates standard time-series measurements.
    Function that calculates the time-series measurements for HeartPy.
    Parameters
    ----------
    rr_list : 1d list or array
        list or array containing peak-peak intervals
    rr_diff : 1d list or array
        list or array containing differences between adjacent peak-peak intervals
    rr_sqdiff : 1d list or array
        squared rr_diff
    measures : dict
        dictionary object used by heartpy to store computed measures. Will be created
        if not passed to function.
    working_data : dict
        dictionary object that contains all heartpy's working data (temp) objects.
        will be created if not passed to function
    Returns
    -------
    working_data : dict
        dictionary object that contains all heartpy's working data (temp) objects.
    measures : dict
        dictionary object used by heartpy to store computed measures.
    Examples
    --------
    Normally this function is called during the process pipeline of HeartPy. It can
    of course also be used separately.
    Assuming we have the following peak-peak distances:
    >>> import numpy as np
    >>> rr_list = [1020.0, 990.0, 960.0, 1000.0, 1050.0, 1090.0, 990.0, 900.0, 900.0, 950.0, 1080.0]
    we can then compute the other two required lists by hand for now:
    >>> rr_diff = np.diff(rr_list)
    >>> rr_sqdiff = np.power(rr_diff, 2)
    >>> wd, m = calc_ts_measures(rr_list, rr_diff, rr_sqdiff)
    All output measures are then accessible from the measures object through
    their respective keys:
    >>> print('%.3f' %m['bpm'])
    60.384
    >>> print('%.3f' %m['rmssd'])
    67.082
    '''

    measures['bpm'] = 60000 / np.mean(rr_list)
    measures['ibi'] = np.mean(rr_list)

    measures['sdnn'] = np.std(rr_list)
    measures['sdsd'] = np.std(rr_diff)
    measures['rmssd'] = np.sqrt(np.mean(rr_sqdiff))
    nn20 = rr_diff[np.where(rr_diff > 20.0)]
    nn50 = rr_diff[np.where(rr_diff > 50.0)]
    # working_data['nn20'] = nn20
    # working_data['nn50'] = nn50
    try:
        measures['pnn20'] = float(len(nn20)) / float(len(rr_diff))
    except:
        measures['pnn20'] = np.nan
    # measures['hr_mad'] = MAD(rr_list)

    return working_data, measures