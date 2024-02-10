from model_operations import Model_Operator

if __name__ == '__main__':
    # Create a model that takes glucose
    op = Model_Operator(128, 8, 0.001, 500, 512, 'lstm', 'glcse')
    # Create a model that takes glucose and time
    op1 = Model_Operator(128, 8, 0.001, 500, 512, 'lstm', 'glcse_hr')
    # Create a model that takes glucose and time and weekday
    op2 = Model_Operator(128, 8, 0.001, 500, 512, 'lstm', 'glcse_hr_wdy')

    # Train the model
    op.train_model()
    # Test the trained model
    op.test_model()

    # Run the trained model on inference
    email = "YOUR E-MAIL-ADRESS"
    password = "YOUR PWD"
    op.plot_inference(email, password)