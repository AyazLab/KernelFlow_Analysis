import os
import datetime
import pandas as pd
import numpy as np
from statistics import mean

class Data_Functions():
    def get_exp_order(self) -> list:
        """Returns the experiment order"""

        exp_order_filename = f"{self.par_ID}_experiment_order.txt"
        exp_order_filepath = os.path.join(self.par_dir, exp_order_filename)

        with open(exp_order_filepath) as f:
            lines = f.readlines()

        exp_order = []
        for line in lines:
            if "Block" in line or "-" in line or line == "\n":
                pass
            else:
                exp_order.append(line.strip("\n"))
        
        return exp_order

    def parse_log_file(self, par_dir, exp_name) -> list:
        """Returns a list of the marker data parsed from the log file"""
        def _parse_udp(udp) -> dict:
            """Returns a dict of the parsed UDP marker information"""

            marker_ID_info = udp[2].strip(",").split("=")
            marker_ID_str = marker_ID_info[0]
            marker_ID = marker_ID_info[1]

            marker_val_info = udp[3].strip(",").split("=")
            marker_val_str = marker_val_info[0]
            marker_val = marker_val_info[1]

            marker_string_info = udp[4].strip(",").split("=")
            marker_string_str = marker_string_info[0]
            marker_string = marker_string_info[1]

            marker_ts_info = udp[5].strip("\n").split("=")
            marker_ts_str = marker_ts_info[0]
            marker_ts = marker_ts_info[1]

            marker_data = {marker_ID_str: marker_ID, marker_val_str: marker_val, marker_string_str: marker_string, marker_ts_str: marker_ts}
            
            return marker_data

        log_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(log_dir):
            if ".log" in filename:
                log_filename = filename
        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath) as f:
            lines = f.readlines()

        udp_lines = []
        for line in lines:
            if "UDP" in line:
                udp_lines.append(line.split("\t")[-1])

        marker_data = []
        try:
            start_udp = udp_lines[0].split(" ")
            marker_data.append(_parse_udp(start_udp))
        except:
            marker_data.append("_")
        try:
            end_udp = udp_lines[1].split(" ")
            marker_data.append(_parse_udp(end_udp))
        except:
            if exp_name == "go_no_go":
                marker_ID = int(marker_data[0]["marker_ID"]) + 1
                marker_val = 22
                marker_string = "go_no_go_end"
                end_ts = marker_data[0]["timestamp"]
                end_ts = int(float(end_ts) + float(lines[-1].split("\t")[0])*1e9 - 0.4*1e9)
                marker_data.append({'marker_ID': marker_ID, 'marker_value': marker_val, 'marker_string': marker_string, 'timestamp': end_ts})
            else:
                marker_data.append("_")

        return marker_data

    def parse_narrative_log_file(self, par_dir, exp_name):
        log_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(log_dir):
            if ".log" in filename:
                log_filename = filename
        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath) as f:
            lines = f.readlines()

        udp_lines = []
        for line in lines:
            if "UDP" in line:
                udp_lines.append(line.split("\t")[0])

        end_time = float(udp_lines[1])*1e9

        return end_time

    def parse_task_order_file(self, par_dir, exp_name) -> pd.DataFrame:
        """Returns a data frame of the task order parsed from the task order file"""

        exp_dir = os.path.join(par_dir, exp_name)
        for filename in os.listdir(exp_dir):
            if ".csv" in filename:
                task_order_filename = filename
        task_order_filepath = os.path.join(exp_dir, task_order_filename)
        task_order = pd.read_csv(task_order_filepath)

        return task_order

    def get_data_filepath(self, par_dir, exp_name) -> str:
        """Returns the filepath"""

        data_dir = os.path.join(par_dir, exp_name, "data")
        for filename in os.listdir(data_dir):
            if ".csv" in filename:
                data_filename = filename
        data_filepath = os.path.join(data_dir, data_filename)

        return data_filepath

    def csv_to_df(self, filepath):
        df = pd.read_csv(filepath)

        return df

    def get_all_marker_timestamps(self, par_dir, exp_order):
        all_marker_timestamps = {}
        for exp_name in exp_order:
            start_marker, end_marker = self.parse_log_file(par_dir=par_dir, exp_name=exp_name)
            if "narrative" in exp_name and "participant_01" not in par_dir and "participant_02" not in par_dir and "participant_03" not in par_dir:
                try:
                    start_ts = start_marker["timestamp"]
                except:
                    start_ts = "_"
                    print("ERROR: start marker not found", exp_name)
                try:
                    end_ts = float(start_ts) + self.parse_narrative_log_file(par_dir, exp_name)
                except:
                    print("ERROR: end marker not found", exp_name)
            else:
                try:
                    start_ts = start_marker["timestamp"]
                except:
                    start_ts = "_"
                    print("ERROR: start marker not found", exp_name)
                try:
                    end_ts = end_marker["timestamp"]
                except:
                    end_ts = "_"
                    print("ERROR: end marker not found", exp_name)
            all_marker_timestamps[exp_name] = [start_ts, end_ts]

        return all_marker_timestamps

    def get_cols(self, df, cols):
        return df[cols]

    def create_col(self, x, num_rows):  # TODO, specify dtype in function call
        return pd.Series([x]*num_rows)

    def flatten(self, input_list):
        return [x for xs in input_list for x in xs]

    def parse_df(self, df, num_blocks, num_trials):
        df_by_block = {}
        for i in range(num_blocks):
            block_name = f"block_{i+1}"
            if i == 0:
                temp_df = df.iloc[i*num_trials:(i+1)*num_trials]
                df_no_nan = temp_df.copy()
            else:
                temp_df = df.iloc[(i*num_trials)+i:((i+1)*num_trials)+i]  # skip Nan line between blocks
                df_no_nan = pd.concat([df_no_nan, temp_df])
            df_by_block[block_name] = temp_df

        return df_by_block, df_no_nan

    def get_exp_ts(self, df, exp_name):
        df_temp = df[df["exp_name"] == exp_name]
        start_ts = df_temp["start_timestamp"].item()
        end_ts = df_temp["end_timestamp"].item()

        return start_ts, end_ts

    def get_exp_dt(self, df, exp_name):
        df_temp = df[df["exp_name"] == exp_name]
        start_dt = datetime.datetime.fromtimestamp(df_temp["start_timestamp"].item()/1e9)
        end_dt = datetime.datetime.fromtimestamp(df_temp["end_timestamp"].item()/1e9)

        return start_dt, end_dt

    def get_start_index_dt(self, df, start_dt):
        try:
            for loc, dt in enumerate(df["datetime"]):
                if not dt < start_dt:
                    break
            if loc < df["datetime"].shape[0]-1:
                return loc
            else:
                print("ERROR: start index datetime not found!")
                return None
        except:
            print("ERROR: start index datetime not found!")
            return None

    def get_end_index_dt(self, df, end_dt):
        try:
            for loc, dt in enumerate(df["datetime"]):  
                if dt > end_dt:
                    break
            return loc
        except:
            print("ERROR: end index datetime not found!")
            return None

    def get_start_index_ts(self, df, start_ts):
        try:
            for loc, ts in enumerate(df["timestamps"]):
                if not ts < start_ts:
                    break
            if loc < df["timestamps"].shape[0]-1:
                return loc
            else:
                print("ERROR: start index timestamp not found!")
                return None
        except:
            print("ERROR: start index timestamp not found!")
            return None

    def get_end_index_ts(self, df, end_ts):
        try:    
            for loc, ts in enumerate(df["timestamps"]):  
                if ts > end_ts:
                    break
            return loc
        except:
            print("ERROR: end index timestamp not found!")
            return None

    def adjust_df_ts(self, df, start_ts, cols):
        df = df.copy()
        for col in cols:
            df[col] = df[col] + start_ts
        return df

    def c_to_f(self, temp):
        """Convert celsius to fahrenheit"""
        
        return round(temp * 9/5 + 32, 2)

class Audio_Narrative(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "audio_narrative"
        self.num_blocks = 1
        self.num_trials = 1
        self.clip_duration = 423  # seconds
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)
        self.df = self.csv_to_df(filepath=self.data_filepath)

        cols = ["pieman_clip.started", "participant_response.text"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp.insert(1, "pieman_clip.ended", self.df_simp["pieman_clip.started"] + self.clip_duration)

        self.response = self.get_response()
        self.block_start_time = self.get_clip_start_time()

    def get_response(self):
        try:
            return self.df_simp["participant_response.text"][0]
        except:
            print("ERROR: Pieman - Need to combine 'participant response' columns into a single column.")

    def get_clip_start_time(self):
        return self.df_simp["pieman_clip.started"][0]

class Go_No_Go(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "go_no_go"
        self.num_blocks = 4
        self.num_trials = 20
        # NOTE: stimulus duration depends on participant response
        # NOTE: inter-stimulus duration is variable
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)

        self.task_order = self.parse_task_order_file(par_dir=par_dir, exp_name=self.exp_name)
        self.task_order_simp = self._simp_task_order(task_order=self.task_order)

        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["match", "GNG_stim", "inter_stim_plus.started", "go_image.started", "go_resp.corr", "go_resp.rt"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp = self.df_simp.copy()
        self.df_simp["GNG_stim"] = self.df_simp["GNG_stim"].apply(self._strip_stim)
        self.df_simp.insert(2, "inter_stim_plus.ended", self.df_simp["inter_stim_plus.started"] + self.df["inter_stim_interval"])
        self.df_simp.insert(4, "go_image.ended", self.df_simp["go_image.started"] + self.df_simp["go_resp.rt"])
        self.df_by_block, self.df_no_nan = self.parse_df(df=self.df_simp, num_blocks=self.num_blocks, num_trials=self.num_trials) 
    
        self._correct_responses(df_by_block=self.df_by_block)
        self._response_times(df_by_block=self.df_by_block)

    def _simp_task_order(self, task_order):
        task_order = task_order["task_order"].to_list()
        task_order_simp = [task.split("_")[0] for task in task_order]

        return task_order_simp

    def _strip_stim(self, row):
        try:
            return row.split("\\")[0].split("_stimuli")[0]
        except: 
            return row  

    def _correct_responses(self, df_by_block):
        self.num_corr_go_list = []
        self.num_corr_gng_list = []

        for i, block in enumerate(df_by_block.values()):
            if (i+1) % 2 != 0:  # Go blocks
                num_corr_go = int(sum(block["go_resp.corr"]))
                self.num_corr_go_list.append(num_corr_go)
            else:  # Go/No-Go blocks
                num_corr_gng = int(sum(block["go_resp.corr"]))
                self.num_corr_gng_list.append(num_corr_gng)

        self.total_corr_go = sum(self.num_corr_go_list)
        self.total_corr_gng = sum(self.num_corr_gng_list)
        self.total_corr = self.total_corr_go + self.total_corr_gng

    def _response_times(self, df_by_block):
        self.resp_time_go_list = []
        self.resp_time_gng_list = []

        for i, block in enumerate(df_by_block.values()):
            if (i+1) % 2 != 0:  # Go blocks
                try:  # handle Nan 
                    resp_time_go = np.nanmean(block["go_resp.rt"])
                    self.resp_time_go_list.append(resp_time_go)
                except:
                    pass
            else:  # Go/No-Go blocks
                try:  # handle Nan
                    resp_time_gng = np.nanmean(block["go_resp.rt"])
                    self.resp_time_gng_list.append(resp_time_gng)
                except:
                    pass

        self.avg_resp_time_go = mean(self.resp_time_go_list)
        self.avg_resp_time_gng = mean(self.resp_time_gng_list)
        self.avg_resp_time = [self.avg_resp_time_go, self.avg_resp_time_gng]

class King_Devick(Data_Functions):  
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "king_devick"
        self.num_blocks = 1
        self.num_trials = 3
        self.num_trials_new = 4  # added a card for participants 15+
        # NOTE: card duration depends on participant response
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)
        
        self.task_order = ["card_1", "card_2", "card_3"]
        self.task_order_new = ["card_1", "card_2", "card_3", "card_4"]  # added a card for participants 15+
        
        self.df = self.csv_to_df(filepath=self.data_filepath)
        num_incorrect_col = pd.Series(self._parse_data_file(par_dir=par_dir))
        self.df.insert(len(self.df.columns), "num_incorrect", num_incorrect_col)
        cols = ["card_image.started", "card_resp.rt", "num_incorrect"]
        self.df_simp = self.get_cols(self.df, cols)
        self.df_simp.insert(1, "card_image.ended", self.df_simp["card_image.started"] + self.df_simp["card_resp.rt"])

    def _parse_data_file(self, par_dir):
        data_dir = os.path.join(par_dir, self.exp_name, "data")
        for filename in os.listdir(data_dir):
            if "data.txt" in filename:
                data_filename = filename
        data_filepath = os.path.join(data_dir, data_filename)

        with open(data_filepath) as f:
            lines = f.readlines()

        par_resp = []
        for line in lines:
            if "number incorrect" in line:
                par_resp.append(line.split(" ")[-1].strip("\n"))

        return par_resp

class N_Back(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "n_back"
        self.num_blocks = 9
        self.num_trials = 20
        self.stim_duration = 0.5  # seconds
        self.inter_stim_duration = 3.0  # seconds
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)
        
        self.task_order = self.parse_task_order_file(par_dir=par_dir, exp_name=self.exp_name)
        self.task_order_simp, self.task_order_simp2 = self._simp_task_order(task_order=self.task_order)

        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["match", "stim_text.started", "stim_resp.corr", "stim_resp.rt"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp.insert(2, "stim_text.ended", self.df_simp["stim_text.started"] + 0.5)     # stimulus duration is 0.5 seconds 
        self.df_simp.insert(3, "inter_stim.started", self.df_simp["stim_text.started"] + 0.5)  # inter-stimulus interval starts 0.5 seconds after stimulus 
        self.df_simp.insert(4, "inter_stim.ended", self.df_simp["inter_stim.started"] + 3)     # inter-stimulus duration is 3 seconds
        self.df_by_block, self.df_no_nan = self.parse_df(df=self.df_simp, num_blocks=self.num_blocks, num_trials=self.num_trials)

        self._correct_responses(df_by_block=self.df_by_block)
        self._response_times(df_by_block=self.df_by_block)

    def _correct_responses(self, df_by_block):
        self.num_corr_ZB_list = []
        self.num_corr_OB_list = []
        self.num_corr_TB_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if "0_back" in task_type:
                num_corr_ZB = int(sum(block["stim_resp.corr"]))
                self.num_corr_ZB_list.append(num_corr_ZB)
            elif "1_back" in task_type:
                num_corr_OB = int(sum(block["stim_resp.corr"]))
                self.num_corr_OB_list.append(num_corr_OB)
            elif "2_back" in task_type:
                num_corr_TB = int(sum(block["stim_resp.corr"]))
                self.num_corr_TB_list.append(num_corr_TB)

        self.total_corr_ZB = sum(self.num_corr_ZB_list)
        self.total_corr_OB = sum(self.num_corr_OB_list)
        self.total_corr_TB = sum(self.num_corr_TB_list)
        self.total_corr = self.total_corr_ZB + self.total_corr_OB + self.total_corr_TB

    def _response_times(self, df_by_block):
        self.resp_time_ZB_list = []
        self.resp_time_OB_list = []
        self.resp_time_TB_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if "0_back" in task_type:
                resp_time_ZB = np.nanmean(block["stim_resp.rt"])
                self.resp_time_ZB_list.append(resp_time_ZB)
            elif "1_back" in task_type:
                resp_time_OB = np.nanmean(block["stim_resp.rt"])
                self.resp_time_OB_list.append(resp_time_OB)
            elif "2_back" in task_type:
                resp_time_TB = np.nanmean(block["stim_resp.rt"])
                self.resp_time_TB_list.append(resp_time_TB)

        self.avg_resp_time_ZB = mean(self.resp_time_ZB_list)
        self.avg_resp_time_OB = mean(self.resp_time_OB_list)
        self.avg_resp_time_TB = mean(self.resp_time_TB_list)
        self.avg_resp_time = [self.avg_resp_time_ZB, self.avg_resp_time_OB, self.avg_resp_time_TB]

    def _simp_task_order(self, task_order):
        task_order = task_order["task_order"].to_list()
        task_order_simp = []
        task_order_simp2 = []
        for task in task_order:
            if "ZB" in task:
                temp = task.split("-")
                task_simp = f"0_back-{temp[1]}"
                task_simp2 = "0_back"
            elif "OB" in task:
                task_simp = "1_back"
                task_simp2 = "1_back"
            elif "TB" in task:
                task_simp = "2_back"
                task_simp2 = "2_back"
            task_order_simp.append(task_simp)
            task_order_simp2.append(task_simp2)
        
        return task_order_simp, task_order_simp2

class Resting_State(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "resting_state"
        self.num_blocks = 2
        self.num_trials = 1
        self.session_duration = 210  # seconds for each task (eyes open and eyes closed)
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)

        self.task_order = self.parse_task_order_file(par_dir=par_dir, exp_name=self.exp_name)
        self.task_order_simp = self._simp_task_order(task_order=self.task_order)
        
        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["trial_cross.started", "halfway_tone.started", "done_sound.started"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)

    def _simp_task_order(self, task_order):
        task_order = task_order["task_order"].to_list()
        task_order_simp = [task.replace(' ', '_') for task in task_order]

        return task_order_simp

class Tower_of_London(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "tower_of_london"
        self.num_blocks = 6
        self.num_trials = 6
        self.stim_duration = 7  # seconds
        self.response_duration = 3  # seconds
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)
        
        self.task_order = self.parse_task_order_file(par_dir=par_dir, exp_name=self.exp_name)
        self.task_order_simp = self._simp_task_order(task_order=self.task_order)

        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["match", "stim_image.started", "stim_text.started", "stim_resp.corr", "stim_resp.rt"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp.insert(2, "stim_image.ended", self.df_simp["stim_image.started"] + self.stim_duration)
        self.df_simp.insert(4, "stim_text.ended", self.df_simp["stim_text.started"] + self.response_duration)
        self.df_by_block, self.df_no_nan = self.parse_df(df=self.df_simp, num_blocks=self.num_blocks, num_trials=self.num_trials) 

        self._correct_responses(df_by_block=self.df_by_block)
        self._response_times(df_by_block=self.df_by_block)

    def _correct_responses(self, df_by_block):
        self.num_corr_MM_list = []
        self.num_corr_ZM_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if task_type == "MM":
                num_corr_MM = int(sum(block["stim_resp.corr"]))
                self.num_corr_MM_list.append(num_corr_MM)
            elif task_type == "ZM":
                num_corr_ZM = int(sum(block["stim_resp.corr"]))
                self.num_corr_ZM_list.append(num_corr_ZM)

        self.total_corr_MM = sum(self.num_corr_MM_list)
        self.total_corr_ZM = sum(self.num_corr_ZM_list)
        self.total_corr = self.total_corr_MM + self.total_corr_ZM

    def _response_times(self, df_by_block):
        self.resp_time_MM_list = []
        self.resp_time_ZM_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if task_type == "MM":
                resp_time_MM = np.nanmean(block["stim_resp.rt"])
                self.resp_time_MM_list.append(resp_time_MM)
            elif task_type == "ZM":
                resp_time_ZM = np.nanmean(block["stim_resp.rt"])
                self.resp_time_ZM_list.append(resp_time_ZM)

        self.avg_resp_time_MM = mean(self.resp_time_MM_list)
        self.avg_resp_time_ZM = mean(self.resp_time_ZM_list)
        self.avg_resp_time = [self.avg_resp_time_MM, self.avg_resp_time_ZM]

    def _simp_task_order(self, task_order):
        task_order = task_order["task_order"].to_list()
        task_order_simp = [task.split("_")[0] for task in task_order]

        return task_order_simp

class Video_Narrative_CMIYC(Data_Functions): 
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "video_narrative_cmiyc"
        self.num_blocks = 1
        self.num_trials = 1
        self.clip_duration = 300  # seconds
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)
        self.df = self.csv_to_df(filepath=self.data_filepath)

        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["video_start.started", "catchme_participant_response.text"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp.insert(1, "video_start.ended", self.df_simp["video_start.started"] + self.clip_duration)

        self.response = self.get_response()
        self.block_start_time = self.get_clip_start_time()

    def get_response(self):
        try:
            return self.df_simp["catchme_participant_response.text"][0]
        except:
            print("ERROR: Catchme - Need to combine 'participant response' columns into a single column.")

    def get_clip_start_time(self):
        return self.df_simp["video_start.started"][0]

class Video_Narrative_Sherlock(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "video_narrative_sherlock"
        self.num_blocks = 1
        self.num_trials = 1
        self.clip_duration = 300  # seconds
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)

        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["video_start.started", "sherlock_participant_response.text"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self.df_simp.insert(1, "video_start.ended", self.df_simp["video_start.started"] + self.clip_duration)

        self.response = self.get_response()
        self.block_start_time = self.get_clip_start_time()

    def get_response(self):
        try:
            return self.df_simp["sherlock_participant_response.text"][0]
        except:
            print("ERROR: Sherlock - Need to combine 'participant response' columns into a single column.")

    def get_clip_start_time(self):
        return self.df_simp["video_start.started"][0]

class vSAT(Data_Functions):
    def __init__(self, par_dir):
        super().__init__()
        self.exp_name = "vSAT"
        self.num_blocks = 4
        self.num_trials = 30
        #NOTE: stimulus duration is variable 
        #NOTE: stimulus location is variable 
        #NOTE: inter-stimulus duration is variable 
        self.data_filepath = self.get_data_filepath(par_dir=par_dir, exp_name=self.exp_name)
        self.marker_data = self.parse_log_file(par_dir=par_dir, exp_name=self.exp_name)

        self.task_order = self.parse_task_order_file(par_dir=par_dir, exp_name=self.exp_name)
        self.task_order_simp = self._simp_task_order(task_order=self.task_order)
 
        self.df = self.csv_to_df(filepath=self.data_filepath)
        cols = ["match", "stim_time", "inter_stim_time", "x_pos", "y_pos", "inter_stim_text.started", "vSAT_square.started", "stim_resp.corr", "stim_resp.rt", "feedback_sound.started"]
        self.df_simp = self.get_cols(df=self.df, cols=cols)
        self._add_pos_col()
        self.df_simp.insert(5, "inter_stim_text.ended", self.df_simp["inter_stim_text.started"]+self.df_simp["inter_stim_time"])  
        self.df_simp.insert(7, "vSAT_square.ended", self.df_simp["vSAT_square.started"]+self.df_simp["stim_time"])  
        self.df_simp.insert(11, "feedback_sound.ended", self.df_simp["feedback_sound.started"]+0.5)  # 0.5 second feedback sound  
        self.df_by_block, self.df_no_nan = self.parse_df(df=self.df_simp, num_blocks=self.num_blocks, num_trials=self.num_trials) 

        self._correct_responses(df_by_block=self.df_by_block)
        self._response_times(df_by_block=self.df_by_block)

    def _correct_responses(self, df_by_block):
        self.num_corr_SAT_list = []
        self.num_corr_vSAT_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if task_type == "SAT":
                num_corr_SAT = int(sum(block["stim_resp.corr"]))
                self.num_corr_SAT_list.append(num_corr_SAT)
            elif task_type == "vSAT":
                num_corr_vSAT = int(sum(block["stim_resp.corr"]))
                self.num_corr_vSAT_list.append(num_corr_vSAT)

        self.total_corr_SAT = sum(self.num_corr_SAT_list)
        self.total_corr_vSAT = sum(self.num_corr_vSAT_list)
        self.total_corr = self.total_corr_SAT + self.total_corr_vSAT

    def _response_times(self, df_by_block):
        self.resp_time_SAT_list = []
        self.resp_time_vSAT_list = []

        for task_type, block in zip(self.task_order_simp, df_by_block.values()):
            if task_type == "SAT":
                resp_time_SAT = np.nanmean(block["stim_resp.rt"])
                self.resp_time_SAT_list.append(resp_time_SAT)
            elif task_type == "vSAT":
                resp_time_vSAT = np.nanmean(block["stim_resp.rt"])
                self.resp_time_vSAT_list.append(resp_time_vSAT)

        self.avg_resp_time_SAT = mean(self.resp_time_SAT_list)
        self.avg_resp_time_vSAT = mean(self.resp_time_vSAT_list)
        self.avg_resp_time = [self.avg_resp_time_SAT, self.avg_resp_time_vSAT]

    def _simp_task_order(self, task_order):
        task_order = task_order["task_order"].to_list()
        task_order_simp = [task.split("_")[0] for task in task_order]

        return task_order_simp

    def _add_pos_col(self):
        x_pos_col = self.df_simp["x_pos"]
        y_pos_col = self.df_simp["y_pos"]

        pos_list = []
        for x_pos, y_pos in zip(x_pos_col, y_pos_col):
            if x_pos == 0 and y_pos == 0:
                pos = "center"
                pos_list.append(pos)
            elif x_pos == 0.25 and y_pos == 0.25:
                pos = "top-right"
                pos_list.append(pos)
            elif x_pos == 0.25 and y_pos == -0.25:
                pos = "bottom-right"
                pos_list.append(pos)
            elif x_pos == -0.25 and y_pos == 0.25:
                pos = "top-left"
                pos_list.append(pos)
            elif x_pos == -0.25 and y_pos == -0.25:
                pos = "bottom-left"
                pos_list.append(pos)
            else:
                pos = "Nan"
                pos_list.append(pos)

        self.df_simp.insert(loc=4, column="position", value=pos_list)
        self.df_simp = self.df_simp.drop(columns=["x_pos", "y_pos"])

class Participant_Behav(Data_Functions):
    def __init__(self, par_num):
        super().__init__()
        self.par_num = par_num
        self.par_ID = f"participant_{self.par_num}"
        data_dir = r"C:\Kernel\participants"
        #self.par_dir = os.path.join(os.getcwd(), "participants", self.par_ID)
        self.par_dir = os.path.join(data_dir, self.par_ID)
        
        self.exp_order = self.get_exp_order()
        self.all_marker_timestamps = self.get_all_marker_timestamps(par_dir=self.par_dir, exp_order=self.exp_order)
        self._create_marker_ts_csv()
        self.marker_ts_df = self._create_marker_ts_df()

        self.audio_narrative = Audio_Narrative(par_dir=self.par_dir)
        self.audio_narrative.start_ts = self.get_start_ts("audio_narrative")
        self.audio_narrative.df_adjusted_ts = self.adjust_df_ts(self.audio_narrative.df_simp, self.audio_narrative.start_ts, ["pieman_clip.started", "pieman_clip.ended"])
        
        self.go_no_go = Go_No_Go(par_dir=self.par_dir)
        self.go_no_go.start_ts = self.get_start_ts("go_no_go")
        self.go_no_go.df_adjusted_ts = self.adjust_df_ts(self.go_no_go.df_no_nan, self.go_no_go.start_ts, ["inter_stim_plus.started", "inter_stim_plus.ended", "go_image.started", "go_image.ended"])
        
        self.king_devick = King_Devick(par_dir=self.par_dir)
        self.king_devick.start_ts = self.get_start_ts("king_devick")
        self.king_devick.df_adjusted_ts = self.adjust_df_ts(self.king_devick.df_simp, self.king_devick.start_ts, ["card_image.started", "card_image.ended"])
        
        self.n_back = N_Back(par_dir=self.par_dir)
        self.n_back.start_ts = self.get_start_ts("n_back")
        self.n_back.df_adjusted_ts = self.adjust_df_ts(self.n_back.df_no_nan, self.n_back.start_ts, ["stim_text.started", "stim_text.ended", "inter_stim.started", "inter_stim.ended"])
        
        self.resting_state = Resting_State(par_dir=self.par_dir)
        self.resting_state.start_ts = self.get_start_ts("resting_state")
        self.resting_state.df_adjusted_ts = self.adjust_df_ts(self.resting_state.df_simp, self.resting_state.start_ts, ["trial_cross.started", "halfway_tone.started","done_sound.started"])

        self.tower_of_london = Tower_of_London(par_dir=self.par_dir)
        self.tower_of_london.start_ts = self.get_start_ts("tower_of_london")
        self.tower_of_london.df_adjusted_ts = self.adjust_df_ts(self.tower_of_london.df_no_nan, self.tower_of_london.start_ts, ["stim_image.started", "stim_image.ended", "stim_text.started", "stim_text.ended"])

        self.video_narrative_cmiyc = Video_Narrative_CMIYC(par_dir=self.par_dir)
        self.video_narrative_cmiyc.start_ts = self.get_start_ts("video_narrative_cmiyc")
        self.video_narrative_cmiyc.df_adjusted_ts = self.adjust_df_ts(self.video_narrative_cmiyc.df_simp, self.video_narrative_cmiyc.start_ts, ["video_start.started", "video_start.ended"])
         
        self.video_narrative_sherlock = Video_Narrative_Sherlock(par_dir=self.par_dir)
        self.video_narrative_sherlock.start_ts = self.get_start_ts("video_narrative_sherlock")
        self.video_narrative_sherlock.df_adjusted_ts = self.adjust_df_ts(self.video_narrative_sherlock.df_simp, self.video_narrative_sherlock.start_ts, ["video_start.started", "video_start.ended"])

        self.vSAT = vSAT(par_dir=self.par_dir)
        self.vSAT.start_ts = self.get_start_ts("vSAT")
        self.vSAT.df_adjusted_ts = self.adjust_df_ts(self.vSAT.df_no_nan, self.vSAT.start_ts, ["inter_stim_text.started", "inter_stim_text.ended", "vSAT_square.started", "vSAT_square.ended", "feedback_sound.started", "feedback_sound.ended"])

        self.by_block_ts_df = self._create_by_block_ts_df()

    def _create_marker_ts_csv(self):
        filepath = os.path.join(self.par_dir, f"{self.par_ID}_marker_timestamps.csv", )
        if os.path.exists(filepath):
            pass
        else:
            marker_list = []
            for exp_name, ts_list in self.all_marker_timestamps.items():
                temp_list = []
                temp_list.append(exp_name)
                temp_list.extend(ts_list)
                marker_list.append(temp_list)
            marker_ts_df = pd.DataFrame(marker_list, columns=["exp_name", "start_timestamp", "end_timestamp"])
            marker_ts_df.to_csv(filepath, index=False)

    def _create_marker_ts_df(self):
        marker_ts_filepath = os.path.join(self.par_dir, f"{self.par_ID}_marker_timestamps.csv")
        
        return self.csv_to_df(marker_ts_filepath)

    def get_start_ts(self, exp_name):
        return int(self.all_marker_timestamps[exp_name][0])

    def get_end_ts(self, exp_name):
        return int(self.all_marker_timestamps[exp_name][1])

    def _create_by_block_ts_df(self):
        def format_ts(exp_name):
            start_ts, end_ts = self.get_exp_ts(self.marker_ts_df, exp_name=exp_name)
            return start_ts/1e9, end_ts/1e9

        by_block_ts_df = {}
        for exp_name in self.exp_order:
            block_ts_df = {}
            if exp_name == "audio_narrative":
                start_ts, _ = format_ts(exp_name)
                block_start_time = self.audio_narrative.df_simp["pieman_clip.started"].item()
                block_start_ts = start_ts + block_start_time
                clip_length = 423  # 423 second clip
                block_end_ts = block_start_ts + clip_length
                block_ts_df[(block_start_ts, block_end_ts)] = exp_name
            elif exp_name == "go_no_go":
                start_ts, _ = format_ts(exp_name)
                for block, block_df in zip(self.go_no_go.task_order_simp, self.go_no_go.df_by_block.values()):
                    block_start_time = block_df["inter_stim_plus.started"].iloc[0]
                    block_start_ts = start_ts + block_start_time
                    block_end_time = block_df["go_image.started"].iloc[-1] + 0.5  # image shown for 0.5 seconds
                    block_end_ts = start_ts + block_end_time
                    block_ts_df[(block_start_ts, block_end_ts)] = block
            elif exp_name == "king_devick":
                start_ts, _ = format_ts(exp_name)
                for block, block_start_time, rt in zip(self.king_devick.task_order, self.king_devick.df_simp["card_image.started"].values, self.king_devick.df_simp["card_resp.rt"].values):
                    block_start_ts = start_ts + block_start_time
                    block_end_ts = block_start_ts + rt
                    block_ts_df[(block_start_ts, block_end_ts)] = block
            elif exp_name == "n_back":
                start_ts, _ = format_ts(exp_name)
                for block, block_df in zip(self.n_back.task_order_simp2, self.n_back.df_by_block.values()):
                    block_start_time = block_df["stim_text.started"].iloc[0]
                    block_start_ts = start_ts + block_start_time
                    block_end_time = block_df["stim_text.started"].iloc[-1] + 0.5  # number shown for 0.5 seconds
                    block_end_ts = start_ts + block_end_time
                    block_ts_df[(block_start_ts, block_end_ts)] = block
            elif exp_name == "resting_state":
                start_ts, _ = format_ts(exp_name)
                block_start_time = self.resting_state.df_simp["trial_cross.started"].item()
                block_start_ts = start_ts + block_start_time
                block_end_ts = block_start_ts + (self.resting_state.df_simp["halfway_tone.started"].item() - block_start_time)
                block_ts_df[(block_start_ts, block_end_ts)] = self.resting_state.task_order_simp[0]
                block_start_time = self.resting_state.df_simp["halfway_tone.started"].item()
                block_start_ts = start_ts + block_start_time
                block_end_ts = block_start_ts + (self.resting_state.df_simp["done_sound.started"].item() - block_start_time)
                block_ts_df[(block_start_ts, block_end_ts)] = self.resting_state.task_order_simp[1]
            elif exp_name == "tower_of_london":
                start_ts, _ = format_ts(exp_name)
                for block, block_df in zip(self.tower_of_london.task_order_simp, self.tower_of_london.df_by_block.values()):
                    block_start_time = block_df["stim_image.started"].iloc[0]
                    block_start_ts = start_ts + block_start_time
                    block_end_time = block_df["stim_text.started"].iloc[-1] + 3  # 3 seconds to respond
                    block_end_ts = start_ts + block_end_time
                    block_ts_df[(block_start_ts, block_end_ts)] = block
            elif exp_name == "video_narrative_cmiyc":
                start_ts, _ = format_ts(exp_name)
                block_start_time = self.video_narrative_cmiyc.df_simp["video_start.started"].item()
                block_start_ts = start_ts + block_start_time  # NOTE
                clip_length = 300  # 300 second clip
                block_end_ts = block_start_ts + clip_length
                block_ts_df[(block_start_ts, block_end_ts)] = exp_name
            elif exp_name == "video_narrative_sherlock":
                start_ts, _ = format_ts(exp_name)
                block_start_time = self.video_narrative_sherlock.df_simp["video_start.started"].item()
                block_start_ts = start_ts + block_start_time
                clip_length = 300  # 300 second clip
                block_end_ts = block_start_ts + clip_length
                block_ts_df[(block_start_ts, block_end_ts)] = exp_name
            elif exp_name == "vSAT":
                start_ts, _ = format_ts(exp_name)
                for block, block_df in zip(self.vSAT.task_order_simp, self.vSAT.df_by_block.values()):
                    block_start_time = block_df["inter_stim_text.started"].iloc[0]
                    block_start_ts = start_ts + block_start_time
                    block_end_time = block_df["feedback_sound.started"].iloc[-1] + 0.5  # 0.5 second delay
                    block_end_ts = start_ts + block_end_time
                    block_ts_df[(block_start_ts, block_end_ts)] = block
            by_block_ts_df[exp_name] = block_ts_df

        return by_block_ts_df

    def get_exp(self, exp_name):
            if exp_name == "audio_narrative":
                return self.audio_narrative
            elif exp_name == "go_no_go":
                return self.go_no_go
            elif exp_name == "king_devick":
                return self.king_devick
            elif exp_name == "n_back":
                return self.n_back
            elif exp_name == "resting_state":
                return self.resting_state
            elif exp_name == "tower_of_london":
                return self.tower_of_london
            elif exp_name == "video_narrative_cmiyc":
                return self.video_narrative_cmiyc
            elif exp_name == "video_narrative_sherlock":
                return self.video_narrative_sherlock
            elif exp_name == "vSAT":
                return self.vSAT

    def get_start_dt(self, exp_name):
        return datetime.datetime.fromtimestamp(int(self.all_marker_timestamps[exp_name][0])/1e9)

def create_behav_results_tables(num_pars):
    def get_num_rows(exp, new=False):
        if new: 
            return int(exp.num_blocks * exp.num_trials_new)
        else:    
            return int(exp.num_blocks * exp.num_trials)

    data_fun = Data_Functions()
    audio_df_list = []
    gng_df_list = []
    kd_df_list = []
    n_back_df_list = []
    tol_df_list = []
    video_cmiyc_df_list = []
    video_sherlock_df_list = []
    vsat_df_list = []

    for i in range(num_pars):
        par_num = f"{(i+1):02d}"
        par = Participant_Behav(par_num=par_num)

        # Audio Narrative ----
        exp = par.audio_narrative
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)

        temp_audio_df = pd.DataFrame([exp.response], columns=["response"])
        temp_audio_df.insert(0, "participant", par_num_col)
        audio_df_list.append(temp_audio_df)

        # Go/No-Go -----
        exp = par.go_no_go
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)

        gng_by_block = exp.df_by_block
        temp_block_df = pd.DataFrame() 
        block_df_list = []
        block_list = []
        
        for block, block_df in zip(exp.task_order_simp, gng_by_block.values()):
            temp_block_df = block_df[["GNG_stim", "go_resp.corr", "go_resp.rt"]]
            block_df_list.append(temp_block_df)
            block_list.append([block]*exp.num_trials)
        block_col = pd.Series(data_fun.flatten(block_list))

        temp_gng_df = pd.DataFrame()
        temp_gng_df = pd.concat(block_df_list, axis=0)
        temp_gng_df.reset_index(inplace=True, drop=True)
        temp_gng_df.insert(0, "block", block_col)
        temp_gng_df.insert(0, "participant", par_num_col)
        temp_gng_df.rename(columns={"GNG_stim": "stim", "go_resp.corr": "correct_response", "go_resp.rt": "response_time"}, inplace=True)
        gng_df_list.append(temp_gng_df.copy())

        # King Devick -----
        exp = par.king_devick

        if int(par_num) > 14:  # participants 15+ have a 4th card
            num_rows = get_num_rows(exp=exp, new=True)
            block_col = pd.Series(exp.task_order_new)
        else:
            num_rows = get_num_rows(exp=exp)
            block_col = pd.Series(exp.task_order)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)
        
        temp_kd_df = exp.df_simp[["card_resp.rt", "num_incorrect"]]
        temp_kd_df.insert(0, "block", block_col)
        temp_kd_df.insert(0, "participant", par_num_col)
        temp_kd_df = temp_kd_df.rename(columns={"card_resp.rt": "response_time"})
        kd_df_list.append(temp_kd_df.copy())

        # N-Back -----
        exp = par.n_back
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)

        n_back_by_block = exp.df_by_block
        temp_block_df = pd.DataFrame() 
        block_df_list = []
        block_list = []
        
        for block, block_df in zip(exp.task_order_simp2, n_back_by_block.values()):
            temp_block_df = block_df[["stim_resp.corr", "stim_resp.rt"]]
            block_df_list.append(temp_block_df)
            block_list.append([block]*exp.num_trials)
        block_col = pd.Series(data_fun.flatten(block_list))

        temp_n_back_df = pd.DataFrame()
        temp_n_back_df = pd.concat(block_df_list, axis=0)
        temp_n_back_df.reset_index(inplace=True, drop=True)
        temp_n_back_df.insert(0, "block", block_col)
        temp_n_back_df.insert(0, "participant", par_num_col)
        temp_n_back_df.rename(columns={"stim_resp.corr": "correct_response", "stim_resp.rt": "response_time"}, inplace=True)
        n_back_df_list.append(temp_n_back_df.copy())

        # Tower of London -----
        exp = par.tower_of_london
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)

        tol_by_block = exp.df_by_block
        temp_block_df = pd.DataFrame() 
        block_df_list = []
        block_list = []
        
        for block, block_df in zip(exp.task_order_simp, tol_by_block.values()):
            temp_block_df = block_df[["stim_resp.corr", "stim_resp.rt"]]
            block_df_list.append(temp_block_df)
            block_list.append([block]*exp.num_trials)
        block_col = pd.Series(data_fun.flatten(block_list))

        temp_tol_df = pd.DataFrame()
        temp_tol_df = pd.concat(block_df_list, axis=0)
        temp_tol_df.reset_index(inplace=True, drop=True)
        temp_tol_df.insert(0, "block", block_col)
        temp_tol_df.insert(0, "participant", par_num_col)
        temp_tol_df.rename(columns={"stim_resp.corr": "correct_response", "stim_resp.rt": "response_time"}, inplace=True)
        tol_df_list.append(temp_tol_df.copy())

        # Video Narrative CMIYC ----
        exp = par.video_narrative_cmiyc
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)
        
        temp_video_cmiyc_df = pd.DataFrame([exp.response], columns=["response"])
        temp_video_cmiyc_df.insert(0, "participant", par_num_col)
        video_cmiyc_df_list.append(temp_video_cmiyc_df)

        # Video Narrative Sherlock ----
        exp = par.video_narrative_sherlock
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)
        
        temp_video_sherlock_df = pd.DataFrame([exp.response], columns=["response"])
        temp_video_sherlock_df.insert(0, "participant", par_num_col)
        video_sherlock_df_list.append(temp_video_sherlock_df)

        # vSAT -----
        exp = par.vSAT
        num_rows = get_num_rows(exp=exp)
        par_num_col = data_fun.create_col(par_num, num_rows=num_rows)

        vsat_by_block = exp.df_by_block
        temp_block_df = pd.DataFrame() 
        block_df_list = []
        block_list = []
        
        for block, block_df in zip(exp.task_order_simp, vsat_by_block.values()):
            temp_block_df = block_df[["stim_resp.corr", "stim_resp.rt"]]
            block_df_list.append(temp_block_df)
            block_list.append([block]*exp.num_trials)
        block_col = pd.Series(data_fun.flatten(block_list))

        temp_vsat_df = pd.DataFrame()
        temp_vsat_df = pd.concat(block_df_list, axis=0)
        temp_vsat_df.reset_index(inplace=True, drop=True)
        temp_vsat_df.insert(0, "block", block_col)
        temp_vsat_df.insert(0, "participant", par_num_col)
        temp_vsat_df.rename(columns={"stim_resp.corr": "correct_response", "stim_resp.rt": "response_time"}, inplace=True)
        vsat_df_list.append(temp_vsat_df.copy())

        # Audio Narrative ----
        audio_df = pd.concat(audio_df_list, axis=0)
        audio_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.audio_narrative.exp_name}_behav.csv")
        audio_df.to_csv(audio_filepath, index=False)
        # Go/No-Go -----
        gng_df = pd.concat(gng_df_list, axis=0)
        gng_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.go_no_go.exp_name}_behav.csv")
        gng_df.to_csv(gng_filepath, index=False)
        # King Devick -----
        kd_df = pd.concat(kd_df_list, axis=0)
        kd_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.king_devick.exp_name}_behav.csv")
        kd_df.to_csv(kd_filepath, index=False)
        # N-Back -----
        n_back_df = pd.concat(n_back_df_list, axis=0)
        n_back_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.n_back.exp_name}_behav.csv")
        n_back_df.to_csv(n_back_filepath, index=False)
        # Tower of London -----
        tol_df = pd.concat(tol_df_list, axis=0)
        tol_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.tower_of_london.exp_name}_behav.csv")
        tol_df.to_csv(tol_filepath, index=False)
        # Video Narrative CMIYC ----
        video_cmiyc_df = pd.concat(video_cmiyc_df_list, axis=0)
        video_cmiyc_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.video_narrative_cmiyc.exp_name}_behav.csv")
        video_cmiyc_df.to_csv(video_cmiyc_filepath, index=False)
        # Video Narrative Sherlock ----
        video_sherlock_df = pd.concat(video_sherlock_df_list, axis=0)
        video_sherlock_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.video_narrative_sherlock.exp_name}_behav.csv")
        video_sherlock_df.to_csv(video_sherlock_filepath, index=False)
        # vSAT -----
        vsat_df = pd.concat(vsat_df_list, axis=0)
        vsat_filepath = os.path.join(os.getcwd(), "results/behavioral", f"{par.vSAT.exp_name}_behav.csv")
        vsat_df.to_csv(vsat_filepath, index=False)