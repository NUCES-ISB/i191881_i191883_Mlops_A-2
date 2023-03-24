# MLOps-Assignment-2
We are supposed to construct a CI/CD pipeline for machine learning system by utilizing the
capabilities of the following tools

● Github and Github Actions
● Flask
● Jenkins
● Docker

Following are the steps that were to be followed to complete the given case study. This application
will be hosted on Github with multiple branches (equals to the # of the team members). We are
required to use Github and Github actions for the CI phase. During the CI phase you are
expected to check the quality of the code using appropriate triggers (push, pull request, fork,
etc.).
1. Initially, you are supposed to select a livestream as an input to your system For example.
Pakistan Stock Exchange, Twitter data etc.
2. You are required to train ML model(s) of your choice on the selected live stream.
3. Then the trained models are wrapped in a Flask application. Flask application must work
in two modes,
a. Live dashboard showing live data along with the prediction of your model(s).
Different metrics of the model must also be shown on the dashboard.
b. In the second mode Flask app will act as a service where a user can provide an
instance of the live data to get the accuracy.

4. Delivery phase will be implemented using Jenkins. Jenkins will be connected with your
above mentioned Git repository and appropriate triggers it will be synced. After the
synchronization, Build is initiated and a docker image is created. This image should be
pushed to your (public) registry and also hosted as a web application (as mentioned in the
point no 3).

Bonus:
Send an email to the registered team members on encountering a data drift.
