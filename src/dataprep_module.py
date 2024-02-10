import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from connection import Api_Connection

class Dataprep:
    def __init__(self, data_collection, hist_path = 'hist_data.csv', x_len = 128, y_len = 16):
        self.hist_path = hist_path
        self.norm_min = 40
        self.norm_max = 300
        self.data_collection = data_collection
        self.data_collection_dims = None
        # input len
        self.x_len = x_len
        # output len
        self.y_len = y_len
        self.step = 1
        # define the number of test chunks for data split
        self.num_test_chunks = 1
    def read_hist_data(self):
        # read data
        data = pd.read_csv(self.hist_path,skiprows=[0])
        # transform to timestamp
        data['timestamp'] = pd.to_datetime(data['Gerätezeitstempel'], format='%d-%m-%Y %H:%M')
        return data

    def hist_data_topandas(self):
        data = self.read_hist_data()
        # transform timestamp
        data_out = pd.DataFrame()
        # set timestpamp
        data_out['timestamp'] = data['timestamp']
        # generate continuous timestamp
        data_out['continuous_timestamp'] = data_out['timestamp'].astype(int)
        regularization = data_out['continuous_timestamp'].iloc[1] - data_out['continuous_timestamp'].iloc[0]
        data_out['continuous_timestamp'] = data_out['continuous_timestamp'] / regularization
        data_out['continuous_timestamp'] = data_out['continuous_timestamp'].astype('int32')
        # add information about breaks in sensor data
        data_out['time_break'] = data_out['continuous_timestamp'].diff()
        # split time stamp
        data_out['day'] = data['timestamp'].dt.day
        data_out['day_of_week'] = data['timestamp'].dt.dayofweek
        data_out['month'] = data['timestamp'].dt.month
        data_out['year'] = data['timestamp'].dt.year
        data_out['hour'] = data['timestamp'].dt.hour
        data_out['minute'] = data['timestamp'].dt.minute
        data_out['glucose'] = data['Glukosewert-Verlauf mg/dL']
        # drop missings
        data_out = data_out.dropna()
        # change glucose type from float to Integer
        data_out['glucose'] = data_out['glucose'].astype('int32')
        data_out['glucose_norm'] = data_out['glucose']
        data_out[data_out['glucose_norm']>self.norm_max]=self.norm_max
        data_out[data_out['glucose_norm']<self.norm_min]=self.norm_min
        data_out['glucose_norm'] = data_out['glucose_norm']-self.norm_min
        data_out['glucose_norm'] = data_out['glucose_norm']/(self.norm_max-self.norm_min)

        # One-hot encode 'hour' variable
        df_encoded = pd.get_dummies(data_out['hour'], prefix='hour')
        # Concatenate the one-hot encoded columns to the original DataFrame
        data_out = pd.concat([data_out, df_encoded], axis=1)
        data_out.drop(columns=['hour_300'],inplace=True)

        # One-hot encode 'hour' variable
        df_encoded = pd.get_dummies(data_out['day_of_week'], prefix='day_of_week')
        # Concatenate the one-hot encoded columns to the original DataFrame
        data_out = pd.concat([data_out, df_encoded], axis=1)
        data_out.drop(columns=['day_of_week_300'], inplace=True)
        data_out.reset_index(inplace=True,drop=True)

        return data_out

    def data_tostring(self):
        data = self.hist_data_topandas()
        # transform data to string
        data = data.astype(str)
        return data

    def get_continious_chunks(self):
        chunks = []
        data = self.hist_data_topandas()

        # Find indices where the time difference is greater than the threshold
        break_indices = data.loc[data['time_break'] > 1].index
        # generate chunks of contious data
        if len(break_indices) == 0:
            chunks.append(data)
        else:
            chunks.append(data.loc[:(break_indices[0])-1])

            if len(break_indices) > 1:
                for cnt, br in enumerate(break_indices):

                    if cnt+1 < len(break_indices):
                        chunks.append(data.loc[br:(break_indices[cnt+1] - 1)])
                    else:
                        chunks.append(data.loc[br:])

        return chunks

    def get_train_and_test_data(self):
        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []

        chunks = self.get_continious_chunks()

        if len(chunks) > 1:
            train_chunks_idx = [i for i in range(len(chunks)-self.num_test_chunks)]
            for tci in train_chunks_idx:
                data = chunks[tci]
                data_len = len(data)
                window = 0
                while window < data_len-(self.x_len+self.y_len):
                    train_data_x.append(data[window:window+self.x_len])
                    train_data_y.append(data[window+self.x_len:window+self.x_len+self.y_len])
                    window += self.step
            for tci in range(self.num_test_chunks):

                data = chunks[-(tci+1)]
                data_len = len(data)
                window = 0
                while window < data_len - (self.x_len + self.y_len):
                    test_data_x.append(data[window:window + self.x_len])
                    test_data_y.append(data[window + self.x_len:window + self.x_len + self.y_len])
                    window += self.step
            return train_data_x, train_data_y, test_data_x, test_data_y
        else:
            print('not enough data')
            return None
        return

    def get_stats(self, data):

        glucose_norm_np = np.array([x['glucose_norm'] for x in data], dtype=np.float32)
        glucose_norm_np_max = np.expand_dims(np.array([np.max(x) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_min = np.expand_dims(np.array([np.min(x) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_mean = np.expand_dims(np.array([np.mean(x) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_median = np.expand_dims(np.array([np.median(x) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_q2 = np.expand_dims(np.array([np.quantile(x,0.2) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_q8 = np.expand_dims(np.array([np.quantile(x,0.8) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_fluc = np.expand_dims(np.array([np.sum(abs(np.diff(x))) for x in glucose_norm_np], dtype=np.float32), axis=1)
        glucose_norm_np_fluc = [4 if x>4 else x for x in glucose_norm_np_fluc]
        glucose_norm_np_fluc = np.array([x / 4 for x in glucose_norm_np_fluc], dtype=np.float32)
        glucose_norm_stats_concat = np.concatenate((glucose_norm_np_max,glucose_norm_np_min,glucose_norm_np_mean,glucose_norm_np_median,glucose_norm_np_q2,glucose_norm_np_q8,glucose_norm_np_fluc),axis=1)
        glucose_norm_stats_concat = np.expand_dims(glucose_norm_stats_concat, axis=1)
        glucose_norm_stats_concat = np.repeat(glucose_norm_stats_concat,128,axis=1)
        return glucose_norm_stats_concat

    def get_train_and_test_data_np(self):
        train_data_x, train_data_y, test_data_x, test_data_y = self.get_train_and_test_data()
        glucose_idx = list(train_data_x[0].columns).index('glucose_norm')
        hour_23_idx = list(train_data_x[0].columns).index('hour_23')

        if self.data_collection == 'glcse':
            train_data_x_glucose = [x['glucose_norm'] for x in train_data_x]
            test_data_x_glucose = [x['glucose_norm'] for x in test_data_x]

        elif self.data_collection == 'glcse_hr_wdy':
            train_data_x_glucose = [x.iloc[:, glucose_idx:] for x in train_data_x]
            test_data_x_glucose = [x.iloc[:, glucose_idx:] for x in test_data_x]

        elif self.data_collection == 'glcse_hr':
            train_data_x_glucose = [x.iloc[:, glucose_idx:hour_23_idx+1] for x in train_data_x]
            test_data_x_glucose = [x.iloc[:, glucose_idx:hour_23_idx+1] for x in test_data_x]

        train_data_y_glucose = [x['glucose_norm'] for x in train_data_y]
        test_data_y_glucose = [x['glucose_norm'] for x in test_data_y]

        train_data_x_glucose_np = np.array(train_data_x_glucose, dtype=np.float32)
        train_data_y_glucose_np = np.array(train_data_y_glucose, dtype=np.float32)
        test_data_x_glucose_np = np.array(test_data_x_glucose, dtype=np.float32)
        test_data_y_glucose_np = np.array(test_data_y_glucose, dtype=np.float32)

        if self.data_collection == 'glcse':
            train_data_x_glucose_np = np.expand_dims(train_data_x_glucose_np, axis=2)
            test_data_x_glucose_np = np.expand_dims(test_data_x_glucose_np, axis=2)

        self.data_collection_dims = train_data_x_glucose_np.shape[2]

        glucose_norm_stats_concat_train = self.get_stats(train_data_x)
        glucose_norm_stats_concat_test = self.get_stats(test_data_x)


        test_data_y_time = [x[['hour','minute']].astype(str) for x in test_data_y]

        # time = [x['hour']+':'+x['minute'] for x in test_data_y_time]
        for cnt, dtfrm in enumerate(test_data_y_time):
            test_data_y_time[cnt]['time'] = dtfrm['hour']+':'+dtfrm['minute']
            test_data_y_time[cnt].drop(columns=['hour','minute'], inplace=True)

        return train_data_x_glucose_np, train_data_y_glucose_np, test_data_x_glucose_np, test_data_y_glucose_np, glucose_norm_stats_concat_train, glucose_norm_stats_concat_test, test_data_y_time

    def get_inference_data_np(self, email, password):
        con = Api_Connection(email, password)
        json_data = con.getData()
        all_measurements = json_data['data']['graphData']
        pd_df = pd.DataFrame({'Timestamp': [], 'glucose': []})
        for measurement in all_measurements[-self.x_len:]:
            pd_df.loc[len(pd_df.index)] = [measurement['Timestamp'], measurement['ValueInMgPerDl']]

        pd_df['glucose'] = pd_df['glucose'].astype('int32')
        pd_df['glucose_norm'] = pd_df['glucose']
        pd_df[pd_df['glucose_norm']>self.norm_max]=self.norm_max
        pd_df[pd_df['glucose_norm']<self.norm_min]=self.norm_min
        pd_df['glucose_norm'] = pd_df['glucose_norm']-self.norm_min
        pd_df['glucose_norm'] = pd_df['glucose_norm']/(self.norm_max-self.norm_min)
        pd_df['timestamp'] = pd.to_datetime(pd_df['Timestamp'], format='%m/%d/%Y %I:%M:%S %p')
        pd_df['hour'] = pd_df['timestamp'].dt.hour
        pd_df['minute'] = pd_df['timestamp'].dt.minute
        pd_df['time'] = pd_df['hour'].astype(str) + ':' + pd_df['minute'].astype(str)
        fluctuation_data = self.get_stats([pd_df])
        inference_data = np.expand_dims(np.reshape(pd_df['glucose_norm'].to_numpy(), (self.x_len, 1)), axis=0)
        self.data_collection_dims = inference_data.shape[2]

        return inference_data, fluctuation_data, pd_df['time'].to_numpy()


    def print_chunk(self, idx):
        chunks = self.get_continious_chunks()
        # Create a line plot
        plt.plot(chunks[idx]['timestamp'], chunks[idx]['glucose_norm'])
        # Show the plot
        plt.show()



if __name__=='__main__':
    hist_path = 'hist_data.csv'
    data_collection = 'glcse'
    dt = Dataprep(data_collection, hist_path)

