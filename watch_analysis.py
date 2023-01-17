import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from behav_analysis import Data_Functions, Participant_Behav
    
class Participant_Watch():
    def __init__(self, par_num):
        self.data_fun = Data_Functions()
        self.par_num = par_num
        self.par_ID = f"participant_{self.par_num}"
        data_dir = r"C:\Kernel\participants"
        #self.par_dir = os.path.join(os.getcwd(), "participants", self.par_ID)
        self.par_dir = os.path.join(data_dir, self.par_ID)
        self.par_behav = Participant_Behav(par_num=self.par_num)
        self.exp_order = self.par_behav.exp_order
        self._dir_list = self._get_data_dirs()
        
        self.modalities = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]
        self.modality_df_dict = self._create_modality_df_dict()
        self.marker_ts_df = self.par_behav.marker_ts_df
        self.exp_modality_dict = self._create_exp_modality_dict()

    def _get_data_dirs(self):
        watch_dir = os.path.join(self.par_dir, "watch_data")
        dir_list = []
        for dir_name in os.listdir(watch_dir):
            dir_list.append(os.path.join(watch_dir, dir_name))

        return dir_list

    def _create_modality_df(self, modality):
        df_list = []
        for watch_dir in self._dir_list:
            filepath = os.path.join(watch_dir, modality + ".csv")
            temp_df = pd.read_csv(filepath)
            initial_ts = int(float(temp_df.columns[0]))

            if modality != "IBI":
                samp_freq = int(temp_df.iloc[0][0])
                ts_col = pd.Series([initial_ts + i/samp_freq for i in range(temp_df.size)])
                dt_col = pd.Series([datetime.datetime.fromtimestamp(ts) for ts in ts_col])
                temp_df = temp_df[1:]
                temp_df.insert(loc=0, column="timestamps", value=ts_col)
                temp_df.insert(loc=1, column="datetime", value=dt_col)
                if modality == "ACC":
                    temp_df.rename(columns={temp_df.columns[2]: "accel_x", temp_df.columns[3]: "accel_y", temp_df.columns[4]: "accel_z"}, inplace=True)
                    temp_df["accel_x"] = temp_df["accel_x"]/64
                    temp_df["accel_y"] = temp_df["accel_y"]/64
                    temp_df["accel_z"] = temp_df["accel_z"]/64
                elif modality == "BVP" or modality == "EDA" or modality == "HR":
                    temp_df.rename(columns={temp_df.columns[2]: modality}, inplace=True)    
                elif modality == "TEMP":
                    temp_df.rename(columns={temp_df.columns[2]: "TEMP_C"}, inplace=True)
                    temp_F_col = pd.Series([self.data_fun.c_to_f(temp_C) for temp_C in temp_df["TEMP_C"]])
                    temp_df = temp_df.drop(columns="TEMP_C")
                    temp_df.insert(loc=2, column="TEMP", value=temp_F_col)
            elif modality == "IBI":
                ts_col = temp_df.iloc[:, 0] + initial_ts
                dt_col = pd.Series([datetime.datetime.fromtimestamp(ts) for ts in ts_col])
                temp_df.insert(loc=0, column="timestamps", value=ts_col)
                temp_df.insert(loc=1, column="datetime", value=dt_col)
                temp_df = temp_df.drop(columns=temp_df.columns[2])
                temp_df.rename(columns={temp_df.columns[2]: modality}, inplace=True)    
            df_list.append(temp_df)

        df = pd.concat(df_list, axis=0)
        df.reset_index(inplace=True, drop=True)

        return df

    def _create_modality_df_dict(self):
        modality_df_dict = {}
        for modality in self.modalities:
            modality_df_dict[modality] = self._create_modality_df(modality)
        
        return modality_df_dict

    def _create_exp_modality_dict(self):
        def _get_behav_cols(exp_name, df):  
            exp = self.par_behav.get_exp(exp_name=exp_name)
            num_blocks = exp.num_blocks
            num_rows = df.shape[0]  # num rows for the modality
            df = df.copy()
            df.reset_index(drop=True)
            block_col_list = []
            trial_col_list = []
            
            if exp_name == "audio_narrative" or exp_name == "video_narrative_cmiyc" or exp_name == "video_narrative_sherlock":
                for ts_tuple, value_dict in self.par_behav.by_block_ts_df[exp_name].items():
                    start_ts = ts_tuple[0]
                    end_ts =  ts_tuple[1]
                    start_idx = self.data_fun.get_start_index_ts(df, start_ts)
                    end_idx = self.data_fun.get_end_index_ts(df, end_ts)
                    block = value_dict["block"]
                    trial = value_dict["trial"]
                    if start_idx == None or end_idx == None:
                        pass
                    else:
                        block_col_list.append(self.data_fun.create_col(None, start_idx)) # 0 to start_idx
                        block_col_list.append(self.data_fun.create_col(block, end_idx-start_idx))  # between start/end idx
                        block_col_list.append(self.data_fun.create_col(None, num_rows-end_idx))  # end_idx to -1
                        trial_col_list.append(self.data_fun.create_col(None, start_idx))
                        trial_col_list.append(self.data_fun.create_col(trial, end_idx-start_idx))
                        trial_col_list.append(self.data_fun.create_col(None, num_rows-end_idx))    

            elif exp_name == "go_no_go" or exp_name == "king_devick" or exp_name == "n_back" or exp_name == "resting_state" or exp_name == "tower_of_london" or exp_name == "vSAT":
                for i, (ts_tuple, value_dict) in enumerate(self.par_behav.by_block_ts_df[exp_name].items()):
                    start_ts = ts_tuple[0]
                    end_ts =  ts_tuple[1]
                    start_idx = self.data_fun.get_start_index_ts(df, start_ts)
                    end_idx = self.data_fun.get_end_index_ts(df, end_ts)
                    block = value_dict["block"]
                    trial = value_dict["trial"]
                    if start_idx == None or end_idx == None:
                        pass
                    else:
                        if i == 0:
                            block_col_list.append(self.data_fun.create_col(None, start_idx))
                            block_col_list.append(self.data_fun.create_col(block, end_idx-start_idx))
                            trial_col_list.append(self.data_fun.create_col(None, start_idx))
                            trial_col_list.append(self.data_fun.create_col(trial, end_idx-start_idx))
                        elif i == exp.num_blocks-1:
                            block_col_list.append(self.data_fun.create_col(block, end_idx-start_idx))
                            block_col_list.append(self.data_fun.create_col(None, num_rows-end_idx))
                            trial_col_list.append(self.data_fun.create_col(trial, end_idx-start_idx))
                            trial_col_list.append(self.data_fun.create_col(None, num_rows-end_idx))
                        else:
                            block_col_list.append(self.data_fun.create_col(block, end_idx-start_idx))
                            trial_col_list.append(self.data_fun.create_col(trial, end_idx-start_idx))

            block_col = pd.concat(block_col_list, axis=0, ignore_index=True) 
            trial_col = pd.concat(trial_col_list, axis=0, ignore_index=True)  
            
            return block_col, trial_col

        exp_modality_dict = {}
        for exp_name in self.exp_order:
            start_dt, end_dt = self.data_fun.get_exp_dt(self.marker_ts_df, exp_name=exp_name)  # start/end of an exp
            exp_modality_data_dict = {} 
            for modality, df in self.modality_df_dict.items():
                block_col, trial_col = _get_behav_cols(exp_name, df)  # get block/trial rows for the exp duration
                start_idx = self.data_fun.get_start_index_dt(df=df, start_dt=start_dt)  # exp start idx for this modality
                end_idx = self.data_fun.get_end_index_dt(df=df, end_dt=end_dt)  # exp end idx for this modality
                if start_idx == None or end_idx == None: 
                    exp_modality_data_dict[modality] = None
                else:
                    block_col_sel = block_col.iloc[start_idx:end_idx].reset_index(drop=True)
                    trial_col_sel = trial_col.iloc[start_idx:end_idx].reset_index(drop=True)
                    modality_df = self.modality_df_dict[modality].iloc[start_idx:end_idx]  # get modality rows for the exp duration
                    modality_df = modality_df.reset_index(drop=True)
                    modality_df.insert(0, "trial", trial_col_sel)
                    modality_df.insert(0, "block", block_col_sel)
                    exp_modality_data_dict[modality] = modality_df  # key: modality, value: combined modality/behav DataFrame
            exp_modality_dict[exp_name] = exp_modality_data_dict  # key: exp name, value: combined modality/behav dict with modalities as keys

        return exp_modality_dict

    def _plot_exp_regions(self, ax):
        for exp_name in self.exp_order:
            start_dt, end_dt = self.data_fun.get_exp_dt(self.marker_ts_df, exp_name=exp_name)
            ax.axvline(start_dt, linestyle="dashed", color="k", alpha=0.75)
            ax.axvline(end_dt, linestyle="dashed", color="k", alpha=0.75)
            if exp_name == "audio_narrative":
                ax.axvspan(start_dt, end_dt, color="yellow", alpha=0.4, label="Audio Narrative")
            elif exp_name == "go_no_go":
                ax.axvspan(start_dt, end_dt, color="green", alpha=0.4, label="Go//No-Go")
            elif exp_name == "king_devick":
                ax.axvspan(start_dt, end_dt, color="blue", alpha=0.4, label="King Devick")
            elif exp_name == "n_back":
                ax.axvspan(start_dt, end_dt, color="purple", alpha=0.4, label="N-back")
            elif exp_name == "resting_state":
                ax.axvspan(start_dt, end_dt, color="pink", alpha=0.4, label="Resting State")
            elif exp_name == "tower_of_london":
                ax.axvspan(start_dt, end_dt, color="orange", alpha=0.4, label="Tower of London")
            elif exp_name == "video_narrative_cmiyc":
                ax.axvspan(start_dt, end_dt, color="red", alpha=0.4, label="Video Narrative CMIYC")
            elif exp_name == "video_narrative_sherlock":
                ax.axvspan(start_dt, end_dt, color="olive", alpha=0.4, label="Video Narrative Sherlock")
            elif exp_name == "vSAT":
                ax.axvspan(start_dt, end_dt, color="cyan", alpha=0.4, label="vSAT")

    def plot_modality(self, modality):
        datetime_fmt = mdates.DateFormatter('%H:%M:%S')
        modality_df = self.modality_df_dict[modality]
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        if modality == "ACC":
            ax.plot(modality_df["datetime"], modality_df["accel_x"], color="black")
            ax.plot(modality_df["datetime"], modality_df["accel_y"], color="darkslategray")
            ax.plot(modality_df["datetime"], modality_df["accel_z"], color="darkblue")
            ax.set_ylabel("Acceleration (g)", fontsize=16, color="k")
            ax.set_title("Acceleration", fontsize=20, color="k")
        elif modality == "BVP":
            ax.plot(modality_df["datetime"], modality_df["BVP"], color="k")
            ax.set_ylabel("BVP", fontsize=16, color="k")
            ax.set_title("Photoplethysmograph", fontsize=20, color="k")
        elif modality == "EDA":
            ax.plot(modality_df["datetime"], modality_df["EDA"], color="k")
            ax.set_ylabel("EDA (μS)", fontsize=16, color="k")
            ax.set_title("Electrodermal", fontsize=20, color="k")
        elif modality == "HR":
            ax.plot(modality_df["datetime"], modality_df["HR"], color="k")
            ax.set_ylabel("Heart Rate (BPM)", fontsize=16, color="k")
            ax.set_title("Heart Rate", fontsize=20, color="k")
        elif modality == "IBI":
            ax.plot(modality_df["datetime"], modality_df["IBI"], color="k")
            ax.set_ylabel("Interbeat Interval (seconds)", fontsize=16, color="k")
            ax.set_title("Heart Rate Variability", fontsize=20, color="k")
        elif modality == "TEMP":
            ax.plot(modality_df["datetime"], modality_df["TEMP"], color="k")
            ax.set_ylabel("Temperature (F)", fontsize=16, color="k")
            ax.set_title("Temperature", fontsize=20, color="k")

        ax.set_xlabel("Time", fontsize=16, color="k")
        ax.xaxis.set_major_formatter(datetime_fmt)
        self._plot_exp_regions(ax=ax)
        ax.legend(bbox_to_anchor=(1.0, 0.75), facecolor='white', framealpha=1)

def create_watch_results_tables(num_pars):
    def _create_df(par_list, exp_name, modality):
        temp_df_list = []
        for par in par_list:
            temp_df = pd.DataFrame()
            try:
                temp_df["trial"] = par.exp_modality_dict[exp_name][modality]["trial"]
                temp_df["block"] = par.exp_modality_dict[exp_name][modality]["block"]
                temp_df["timestamps"] = par.exp_modality_dict[exp_name][modality]["timestamps"]
                temp_df[modality] = par.exp_modality_dict[exp_name][modality][modality]
                temp_df.reset_index(inplace=True, drop=True)
                par_num_col = par.data_fun.create_col(par.par_num, num_rows=temp_df.shape[0])
                temp_df.insert(loc=0, column="participant", value=par_num_col) 
            except:
                temp_df[modality] = None
            temp_df_list.append(temp_df)
        df = pd.concat(temp_df_list, axis=0)
        df.reset_index(inplace=True, drop=True)
        return df

    def _data_to_excel(exp_name, data_dict):
        filepath = os.path.join(os.getcwd(), "results", "watch", f"{exp_name}_watch.xlsx")
        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(os.path.dirname(filepath))
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            for modality, df in data_dict.items():
                df.to_excel(writer, sheet_name=modality, index=False)

    par_list = []                
    for i in range(num_pars):
        par_num = f"{(i+1):02d}"
        print("\n-----", par_num, "-----")
        par = Participant_Watch(par_num=par_num)
        par_list.append(par)

    for exp_name in par.exp_order:
        data_dict = {}
        for modality in par.modalities: 
            if modality == "ACC":
                temp_df_list = []
                for par in par_list:
                    temp_df = pd.DataFrame()
                    try:
                        temp_df["trial"] = par.exp_modality_dict[exp_name][modality]["trial"]
                        temp_df["block"] = par.exp_modality_dict[exp_name][modality]["block"]
                        temp_df["timestamps"] = par.exp_modality_dict[exp_name][modality]["timestamps"]
                        temp_df["accel_x"] = par.exp_modality_dict[exp_name][modality]["accel_x"]
                        temp_df["accel_y"] = par.exp_modality_dict[exp_name][modality]["accel_y"]
                        temp_df["accel_z"] = par.exp_modality_dict[exp_name][modality]["accel_z"]
                        temp_df.reset_index(inplace=True, drop=True)
                        par_num_col = par.data_fun.create_col(par.par_num, num_rows=temp_df.shape[0])
                        temp_df.insert(loc=0, column="participant", value=par_num_col)
                    except:
                        temp_df["accel_x"] = None
                        temp_df["accel_y"] = None
                        temp_df["accel_z"] = None
                    temp_df_list.append(temp_df)
                ACC_df = pd.concat(temp_df_list, axis=0)
                ACC_df.reset_index(inplace=True, drop=True)
                data_dict[modality] = ACC_df.dropna()
            elif modality == "BVP" or modality == "EDA" or modality == "HR" or modality == "IBI" or modality == "TEMP":
                modality_df = _create_df(par_list, exp_name, modality)
                data_dict[modality] = modality_df.dropna()

        _data_to_excel(exp_name, data_dict)