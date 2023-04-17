# Inventory Monitoring at Distribution Centers (Counting objects in a bin)

Distribution centers often use robots to move objects as part of their operations. Things
are carried in bins that can contain multiple objects. Accurately tracking inventory and
ensuring that delivery consignments have the correct number of items is crucial for the
smooth operation of the distribution centers. To solve this problem, we propose building a
Classifier ML model that counts the number of objects in each bin (items per bin is always
between 1 and 5). This project will use the Amazon Bin Image Dataset and AWS SageMaker
to preprocess data and train the model. The project will demonstrate end-to-end machine
learning engineering skills acquired during the nano degree.

## Documentaion
For the full documentation, refer to project [report](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/blob/main/docs/proposal.pdf).

### AWS Execution Role:
The AWS execution role used for the project should have the following access:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

# Libraries Used
- [Python 3.x](https://www.python.org)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pytorch](https://pytorch.org/)
## Directory structure
- [docs](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/tree/main/docs): Contains the project [proposal](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/blob/main/docs/proposal.pdf) and [report](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/blob/main/docs/report.pdf).
- [screenshots](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/tree/main/screenshots): Contains important screenshots taken throughout the project.
- [sagemaker.html](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/blob/main/sagemaker.html): Exported Notebook.
- [sagemaker.ipynb](https://github.com/Phoenix6001/udacity-mlnd-capstone-inventory-monitoring-at-distribution-centers/blob/main/sagemaker.ipynb): Python notebook file.
## Project Set Up and Installation
This project was developed and tested AWS SageMaker. It was created from a starter file provided by udacity [here](https://github.com/udacity/nd009t-capstone-starter/tree/master).

## Dataset
### Overview
To accomplish this project we will utilize the [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/), which comprises
500,000 images depicting bins that contain one or multiple objects. Each image in the
Dataset is accompanied by a metadata file containing information such as the number of
objects, dimensions, and the object type. Our objective is to classify the number of objects
present in each bin.

- We used only a tiny subset (10%) of the original Amazon bin image dataset to
avoid large cost.
- The dataset has been split into 80% training, 10% validation, and 10% test
datasets.

### Access
In order to obtain the dataset, one can utilize the Python boto3 library to communicate with the S3 bucket. This particular project employs the SageMaker Processing Job to carry out ETL (Extract, Transform, and Load) procedures on the dataset.
## Model Training

The solution to this problem involves training a deep learning model using the ResNet-50 architecture to predict the object counts in an image from 1 - 5. Using transfer learning with pre-trained Torch models, the ResNet-50 model will be fine-tuned on the given dataset. We trained the model and tuned the hyperparameters using Amazon Sagemaker. The model was evaluated on the validation set using Accuracy metrics. The final model has been deployed and tested on a test set to evaluate its performance.

We chose transfer learning because it has been shown to improve the performance of models on image classification tasks, especially in scenarios with limited training data. By starting with a pre-trained model, we can utilize its learned features and modify them to suit our particular problem domain. This approach can minimize training time and enhance accuracy.

In this project, we focus on tuning two key parameters - the learning_rate and batch_size - as these impact both the model's accuracy and speed of conversion. The `learning_rate` falls within `0.001 to 0.1`, while the `batch_size` can take on one of five values `(32, 64, 128, 256, or 512)`.

### Evaluation
The model was deployed to a Sagemaker endpoint using an ml.m5.large instance to enable inference. The inference script is designed to accept an image URL as input.
The model achieved a 31.5% accuracy.

|       METRIC  |  VALUE  | 
|---------------|---------|
|  Accuracy (%) |  31.5 % |



## Machine Learning Pipeline
Our ML Pipeline for this project consists of the following
- Data preparation: A subset of the Amazon Bin Image Dataset was doenloaded and preprocessed.
- Data split: partition the extracted dataset into train, test, and validation sets, allocating 10% for testing, 10% for validation, and 80% for training
- Amazon SageMaker Hyperparameters Tuner was used to find the best hyperparameters for the model, including the learning rate and batch size.
- Model Training: We employed the best hyperparameters to train our model, utilizing transfer learning of a pre-trained ResNet50 model from PyTorch to accelerate training and enhance accuracy.
- Model Evaluation: The model has been evaluated and achieved a 31.5% accuracy
- Model deployment: The model was deployed using Amazon SageMaker Endpoint.
## Standout Suggestions
### Model Deployment :point_down:
![Model Deployment](./screenshots/Endpoint.png?raw=true "Endpoint deployment")

![query endpoint](./screenshots/inference1.png?raw=true "query endpoint")

### Hyperparameter Tuning
To achieve the best results, we execute a hyperparameter tuning job that selects
parameters from the search space, runs a training job, and then makes predictions. The
primary objective of this process is to improve the Test Loss metric.

> Completed Hyperparameter Tuning Job :point_down:
![Hyperparameter Tuning Job](./screenshots/Hyperparameter-tuning.png?raw=true "Completed Hyperparameter Tuning Job")

> Summery of the best training job :point_down:

![best training job](./screenshots/best-training-job-summary.png?raw=true "best training job")

### Debugging and profiling
We employed the SMDebug client library from Amazon SageMaker to facilitate model debugging and profiling. The Sagemaker debugger allows us to monitor our machine learning model's training performance, record training and evaluation metrics, and plot learning curves. Additionally, it can detect potential problems such as overfitting, overtraining, poor weight initialization, and vanishing gradients.

> Debugging and profiling :point_down:

![Debugging and profiling](./screenshots/training_debug_values.png?raw=true "Debugging and profiling")


### Reduce Cost

We used `t3.2xlarge` EC2 instance, low-cost burstable general-purpose instance types that
provide a baseline level of CPU performance with the ability to burst CPU usage anytime for
as long as required.

> EC2 Instance :point_down:

![EC2 Instance](./screenshots/EC2_Instance_summary_page.png?raw=true "EC2 Instance")

> Cost analysis :point_down:

![Cost analysis](./screenshots/cost_analysis_sagemaker_vs_ec2.png?raw=true "Cost analysis")

### Multi-Instance Training

Amazon SageMaker multi-instance training was used to speed up the training process by distributing the training workload across multiple instances. This can be a useful technique when training large models or datasets.

``` python
# TODO: Train your model on Multiple Instances
###in this cell, create and fit an estimator using multi-instance training
estimator_multi_instance = PyTorch(
    source_dir="scripts",
    entry_point="train.py",
    base_job_name='pytorch-multi-instance-amazon-multi-instance-bin-image-classification',
    role=ROLE,
    instance_count=4,
    instance_type='ml.m5.2xlarge',
    framework_version='1.9',
    py_version='py38',
    hyperparameters=hyperparameters,
    output_path=f"s3://{BUCKET}/output_multi_instance/",
    ## Debugger and Profiler parameters
    rules = rules,
    debugger_hook_config=debugger_config,
    profiler_config=profiler_config,
)
```
