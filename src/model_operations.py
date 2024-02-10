from dataprep_module import Dataprep
from models import LSTM

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Model_Operator:
    def __init__(self, x_len, y_len, learning_rate = 0.0001, num_epochs = 1000,
                 batch_size = 512, model_name = "default", data_collection = "glcse_hr_wdy"):
        self.x_len = x_len
        self.y_len = y_len
        self.norm_min = 40
        self.norm_max = 300
        self.hist_path = 'hist_data.csv'
        self.data_collection = data_collection
        # Hyperparameters
        self.input_size = None
        self.hidden_size = 64
        self.num_layers = 2
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # Updated batch size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self):
        # Set random seed for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

    def load_train_and_test_data(self):
        dt = Dataprep(self.data_collection, self.hist_path, self.x_len, self.y_len)
        train_data_x, train_data_y, test_data_x, test_data_y, fluctuation_train, fluctuation_test, test_data_y_time = dt.get_train_and_test_data_np()
        self.input_size = dt.data_collection_dims
        return train_data_x, train_data_y, test_data_x, test_data_y, fluctuation_train, fluctuation_test, test_data_y_time

    def load_inference_data(self,email, password):
        dt = Dataprep(self.data_collection, self.hist_path, self.x_len, self.y_len)
        inference_data, fluctuation_data, test_data_y_time = dt.get_inference_data_np(email, password)
        self.input_size = dt.data_collection_dims
        return inference_data, fluctuation_data, test_data_y_time

    def init_model(self):
        model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.y_len).to(self.device)
        return model

    def train_model(self):
        self.set_seed()
        train_data_x, train_data_y, test_data_x, test_data_y, fluctuation_train, fluctuation_test, test_data_y_time = self.load_train_and_test_data()

        model = self.init_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        train_data = torch.from_numpy(train_data_x).to(self.device)
        test_data = torch.from_numpy(test_data_x).to(self.device)
        train_labels = torch.from_numpy(train_data_y).to(self.device)
        test_labels = torch.from_numpy(test_data_y).to(self.device)

        fluctuation_train_torch = torch.from_numpy(fluctuation_train).to(self.device)
        fluctuation_test_torch = torch.from_numpy(fluctuation_test).to(self.device)
        loss_list = []
        test_loss_list = []
        epoch_list = []
        # Training the model
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = model(train_data, fluctuation_train_torch)
            loss = criterion(outputs, train_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

                # Test the model
                model.eval()
                with torch.no_grad():
                    test_outputs = model(test_data,fluctuation_test_torch)
                    test_loss = criterion(test_outputs, test_labels)
                    print(f'Test Loss: {test_loss.item():.4f}')

                    # Convert test_outputs to a NumPy array
                    test_outputs_np = test_outputs.cpu().numpy()

                    # Save the NumPy array to a file
                    np.save('test_outputs.npy', test_outputs_np)

                    loss_list.append(loss.tolist())
                    test_loss_list.append(test_loss.tolist())
                    epoch_list.append(epoch)
                model.train()


        plt.plot(epoch_list, loss_list, label='Train Loss', linestyle='--',marker='o')
        plt.plot(epoch_list, test_loss_list, label='Test Loss', linestyle='-',marker='x')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Train vs. Test Loss')
        plt.legend()
        plt.savefig('plots/training_' + str(datetime.datetime.now()) + '.png')
        plt.close()
        torch.save(model.state_dict(), 'models/'+self.model_name+'.pt')

    def test_model(self):
        train_data_x, train_data_y, test_data_x, test_data_y, fluctuation_train, fluctuation_test, test_data_y_time = self.load_train_and_test_data()
        model = self.init_model()
        model.load_state_dict(torch.load('models/'+self.model_name+'.pt'))
        model.eval()
        criterion = nn.MSELoss()
        data = torch.from_numpy(test_data_x).to(self.device)
        labels = torch.from_numpy(test_data_y).to(self.device)

        fluctuation_test_torch = torch.from_numpy(fluctuation_test).to(self.device)
        test_outputs = model(data,fluctuation_test_torch)
        test_loss = criterion(test_outputs, labels)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Convert test_outputs to a NumPy array
        outputs_np = test_outputs.cpu().detach().numpy()

        idx = 0

        # Plot the input and output data
        plt.plot(test_data_y_time[idx]['time'], (test_data_y[idx] * (300 - 40)) + 40, label='Real Data', linestyle='--', marker='o')
        plt.plot(test_data_y_time[idx]['time'], (outputs_np[idx] * (300 - 40)) + 40, label='Model Predictions', linestyle='-', marker='x')

        #  return_sequences=True))

        # Customize the plot
        plt.xlabel('Time')
        plt.ylabel('Glucose (mg/dL)')
        plt.title('Comparison of Input and Test Outputs')
        plt.legend()
        plt.savefig('plots/prediciton_idx:'+str(idx)+'_'+str(datetime.datetime.now())+'.png')

        plt.close()

    def inference(self, email, password):
        inference_data, fluctuation_test, inference_time = self.load_inference_data(email, password)
        model = self.init_model()
        model.load_state_dict(torch.load('models/' + self.model_name + '.pt'))
        model.eval()

        data = torch.from_numpy(inference_data).to(self.device)

        fluctuation_test_torch = torch.from_numpy(fluctuation_test).to(self.device)

        inference_outputs = model(data.float(), fluctuation_test_torch.float())

        return inference_outputs, inference_time, inference_data

    def inference_data(self, email, password):
        inference_outputs, inference_time, inference_data = self.inference(email, password)

        print(np.squeeze(inference_data))
        print(inference_outputs.cpu().detach().numpy()[0])

        for i in range(self.y_len):
            inference_time = np.append(inference_time, ('+' + str((i + 1) * 5) + 'min'))
        print(inference_time)

        data = np.append(np.squeeze(inference_data), inference_outputs.cpu().detach().numpy()[0])
        return inference_time, ((data * (300 - 40)) + 40),inference_time[-self.y_len:], ((data[-self.y_len:] * (300 - 40)) + 40)

    def plot_inference(self, email, password):
        x_all, y_all, x_future, y_future = self.inference_data(email, password)

        plt.plot(x_all, y_all , label='Prediction')

        # Highlight the last 30 values with a different color
        plt.plot(x_future, y_future, color='red', label='Future 16 Values')
        plt.axvline(x=128, color='red', linestyle='--', label='Current Time')
        plt.xticks(np.arange(0, self.y_len+self.x_len, 15))
        plt.xlabel('Time')
        plt.ylabel('Glucose (mg/dL)')
        plt.title('Inference')
        plt.legend()
        plt.savefig('plots/inference.png')

        plt.close()





