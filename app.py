import pandas as pd
import streamlit as st

st.markdown("""
    <style>
    /* Change the color of all h2 header tags */
    h2 {
        color: #489788;
    }
    </style>
    """, unsafe_allow_html=True)


def subheading(text):
    st.markdown(f'<h5 style="color: #595959;">{text}</h5>', unsafe_allow_html=True)

# Define the header sections
def display_section_1():
    st.markdown("<div id='section_1'></div>", unsafe_allow_html=True)
    st.header("Introduction")
    st.write("""
    A hotel, fully booked for peak season, faces last-minute cancellations from guests who booked months in advance. This creates chaos with vacant
    rooms, lost revenue, and scrambling staff. The challenge is to predict which bookings are likely to cancel in advance to prevent this.
    """)
             
    subheading("""Problem Definition:""")
    st.write("""
    - **Who** are the guests most likely to cancel?
    - **What** booking behaviours increase the likelihood of a cancellation?
    - **When** are these cancellations most likely to occur?
    - **Where** do most cancellations come from?
    - **Why** are guests cancelling?
    """)
    subheading("""Proposed Solution:""")
    st.write("""         
    We propose a data-driven approach that uses machine learning and Explainable AI (XAI) to accurately predict 
    hotel reservation cancellations. Our solution not only forecasts which bookings are most likely to cancel but 
    also explains why, helping hotels take proactive measures, reduce financial losses, optimize resource allocation, 
    and enhance guest satisfaction.""")
    subheading("""Project Motivation""")
    st.write("""
    Our project is motivated by the growing unpredictability and financial strain in the hospitality industry caused by
    hotel chains facing high sunken costs from operating above necessary capacity. As the industry grows, mitigating costs
    from sudden cancellations is critical.
    """)
    subheading("""Literature Review:""")
    st.write("""
             
    Research by **Sánchez et al. (2020)** highlights that critical impact that last minute cancellations
    tend to have on hotel revenues. This study leverages Artificial Intelligence techniques to predict which customers are 
    most likely to cancel close to their stay with an accuracy of over 80%.  
                 
    Similarly, **Lin (2023)** uses a combination of random forests and logistic regression to forecaster cancellations. These models
    focus on optimising hotel operations, but like many machine learning approaches, they function as “black boxes” providing little
    to no insight into the decision making processes behind their predictions.  
   
    **Özkurt (2024)** addresses this gap by introducing LIME and SHAP, two prominent XAI techniques, to interpret the predictions of machine learning
    models. By integrating explainable AI into our project, this model will not only contribute significantly to the comprehension of customer churn
    in the hospitality industry but will also offer insights crucial for prevention strategies for hotels
    """)

def display_section_2():
    st.markdown("<div id='section_2'></div>", unsafe_allow_html=True)
    st.header("Dataset Description")
    subheading("""Dataset Link""" )
    st.write("""
        https://www.kaggle.com/datasets/saadharoon27/hotel-booking-dataset
    """)
    subheading("""Dataset Description""" )
    st.write("""    
    We chose the Hotel Booking Dataset from Kaggle for its comprehensive features
    on hotel reservations, including guest demographics, booking details, and cancellation
    statuses. Key features like lead time, previous cancellations, deposit type, and booking
    channels provide insights into booking behavior, helping assess cancellation risk. By focusing
    on these indicators, we aim to identify patterns contributing to cancellations.
    """)


def display_section_3():
    st.markdown("<div id='section_3'></div>", unsafe_allow_html=True)
    st.header("Methods")
    subheading("""We will undertake various steps including, but not limited to:""")
    st.write("""    
    - Remove parameters that do not impact results in any way 
    - Handle missing values (of all columns including Churn column)
    - Assign numerical values to categorical data 
    - Min-Max normalisation for numerical values
    """)
    subheading("""We have identified the following Supervised Learning ML Algorithms/Models for our project:""")
    st.write(""" 
    - Linear regression
    - logistic regression
    - Naive Bayes
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbors
    - Gradient Boosting
    - Extreme Gradient Boosting (XGBoost)
    - LightGBM
    - Adaptive Boosting (AdaBoost)
    - CatBoost

    """)
    subheading("""Gantt Chart:""")
    st.image('ganttChart.png', caption='', use_column_width=True)


def display_section_4():
    st.markdown("<div id='section_4'></div>", unsafe_allow_html=True)
    st.header("Intended Results")
    st.write("""
    Our goal is to develop a model that predicts customer churn, helping hotels manage cancellations, optimize revenue,
    reduce operational costs, and improve guest satisfaction. We focus on reducing resource waste from overbooking or
    under-utilization while ensuring transparency, fairness, and avoiding bias in our model. Previous research has predicted
    hotel churn with success, and we aim for at least 85% accuracy, using Explainable AI to enhance insights. Key metrics for evaluation include:
    """)
    subheading("""Success Metrics: """)
    st.write("""
    - R-Squared Score
    - Mean Squared Error
    - Mean Absolute Error
    - Root Mean Squared Error
    - Accuracy Score
    """)
    subheading("""eXplainable Artificial Intelligence (XAI) metrics:""")
    st.write("""
    - LIME (Local Interpretable Model-agnostic Explanations)
    - SHAP (SHapley Additive exPlanations)
    """)

def display_section_6():
    st.markdown("<div id='section_4'></div>", unsafe_allow_html=True)
    st.header("Midterm Checkpoint")
    subheading("""Methods""")
    st.write("""
    Prior to model training, we undertook the following steps to prepare the raw hotel booking dataset. This step involved multiple rounds of systematic data cleaning, preprocessing, and feature engineering. First, the dataset was thoroughly analysed to evaluate which columns were categorical, numerical, and which columns had the maximum number of missing values/needed cleaning. After that, the data preprocessing was broken down into the following steps:
    - **Data Privacy:** removed sensitive personal information including names, emails, phone numbers, credit card details.
    - **Data Transformation:** we converted the hotel type into binary format, representing ‘City Hotel’ as 0 and ‘Resort Hotel’ as 1. 
    - **Feature Engineering:** Extracted a day of the week (Monday = 0; Sunday = 6). Created a binary feature indicating weekend arrivals.
    - **Missing and invalid values’ data handling:** filled in missing values in the ‘children’ column with the median value. Removed rows with an adult count of zero, as these are considered invalid.
    - **Room Type Change Indicator:** Created a feature to indicate if room type changed between booking and check-in. Dropped original room type columns after creating the indicators.
    - **Encoding for Categorical Variables:** one-hot encoding for standard categorical variables. Frequency encoding for ‘country’, ‘agent’, ‘company’.
    - **Outlier Management:** limited extreme values in ‘adr’ (average daily rate) column to manage outliers effectively. 
    - **Reservation Timing:** computed number of days between last reservation status date and arrival date (temporarily computing by combining arrival day number, month number, and year number). Dropped original reservation status data after calculation. XGBoost does not handle datetime values, and hence the columns split into day number, month number, and year number were kept that way and not combined into a datetime format.
    - **Numerical Standardisation:** standardised to mean=0 and standard deviation=1. This ensures consistency and compatibility across features.
    After the steps above were implemented, the processed dataset was split into training and testing sets (80:20 split) for model training and evaluation.
    """)

    subheading("ML Algorithm Implementation")
    st.write("""
    XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning algorithm. It enhances predictive accuracy by iteratively improving upon decision trees within a gradient-boosting framework. We chose XGBoost as the initial model in our project to predict hotel churn, as it offers flexibility, computational efficiency, and strong performance with classification tasks. Its ability to handle various data types, coupled with its robust design to limit overfitting, made it an ideal choice for identifying bookings most likely to cancel. 
    As a starting point, we initialised XGBoost with its default parameters to get an idea of the model’s default performance on the dataset. The defaults were:
    - learning rate of **0.3** for controlled step sizes
    - maximum tree depth of **6** for balanced complexity
    - **100** boosting rounds to build a comprehensive sequence of trees
    - **subsample** and **colsample_bytree** values set to **1.0**
    - **gamma** set to **0**, allowing unrestricted splitting within trees
    - **min_child_weight** set to **1** to control tree depth with minimum leaf weight.
    Our expectations were that the default would provide reliable initial predictions. However, in reality, these expectations were exceeded as the model was able to capture key patterns of the dataset more effectively than hypothesised. This was a positive indicator for the future steps as it served as a strong baseline, setting an even higher standard for further optimization and accurate hotel churn predictions. 
    """)

    subheading("Important additional preprocessing step: ")
    st.write("""
    Initially, the model achieved an unusually high accuracy of 100%. To investigate this, we examined the correlation between each feature and the target variable (is_cancelled). This revealed that 3 features (status_Canceled, status_Check-Out, and days_since_reservation_status) had a strong correlation with the target. These features were inadvertently providing information about the cancellation status directly, causing data leakage and artificially inflating the model’s performance. Therefore, these 3 columns were dropped.
    """)

    subheading("Quantitative Metrics and Analysis of XGBoost")
    st.image('quant_metrics.png', caption='', use_column_width=True)

    st.write("""
    As mentioned earlier, an 80:20 train: test split was used. First, train and test Accuracy and AUC was evaluated (figure below). We evaluated the model’s effectiveness by using a standard 80-20 train/test split. We then examined both the accuracy and AUC metrics for the training and testing sets.
A high training accuracy of 89.81% and AUC of 0.966 indicates XGBoost’s effective learning on the training data. The test accuracy of 88.43% and an AUC of 0.955 suggests that the model generalises well to unseen data with minimal overfitting. This slight drop in test accuracy and AUC compared to training metrics is expected and is within a reasonable range. All in all, These results imply that the model captures underlying patterns effectively while maintaining robustness when applied to new data.
    """)

    st.write("""
    **A closer look at AUC:** AUC (Area Under the Curve) is a metric used to evaluate the performance of binary classification models. AUC refers to the area under the Receiver Operating Characteristic (ROC) curve, which plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) at various threshold settings. An AUC score closer to 1 demonstrates the model's ability to correctly rank positive samples higher than negative samples. In our case, a high test AUC of 0.955 shows that the model is effective at distinguishing between cancellation and non-cancellations.
    """)

    st.image('acc.png', caption='', use_column_width=True)

    st.write("""
    For class 0, XGBoost achieves high precision (0.90), recall (0.92), and F1-score (0.91), while for class 1, it maintains a precision of 0.86, recall of 0.82, and F1-score of 0.84. The weighted averages across precision, recall, and F1-score (all at 0.88) show us balanced performance across classes. Segmented by class, we can see a flight drop in F1 score from 0.91 (class 0) to 0.84 (class 1). This suggests that further tuning could improve the model’s ability to correctly identify more instances of class 1.

To gain more insight into what features contributed the most to our predictions, we plotted the 10 most important features. This feature importance analysis below led to the conclusion that Lead Time and ADR (Average Daily Rate) emerged as the most influential predictors. This aligns with expectations, since lead time, i.e., the time between booking and check-in often impacts the likelihood of cancellation. ADR is also closely related through revenue potential and booking trends.
    """)

    st.image('fimportance.png', caption='', use_column_width=True)
    st.write("""
    Based on our machine learning algorithm, we deduced various results which were explored through SHAP and LIME, two popular explainable AI techniques that are used for interpreting machine learning models through describing how features contribute to a model’s individual predictions.
    \n**1) SHAP:**
    SHAP is a game-theory-based method that measures how each feature contributes to the final prediction. Features with positive SHAP values increase the probability of cancellation, while those with negative SHAP values decrease it, meaning they reduce the likelihood of cancellation. The **magnitude** of the SHAP value represents how strongly the feature impacts the model's prediction—whether positively or negatively. 
We generated **SHAP summary plots** to visualize how each individual feature affects the probability of cancellation. This visual representation allowed us to clearly see which features were driving cancellations and which were helping reduce them.The summary plot is shown below: 
    """)
    st.image('shap_plot.png', caption='', use_column_width=True)

    st.write("""
    **SHAP Analysis and Interpretations:**
    \nThis SHAP summary plot illustrates the various impacts of features on the likelihood of calculation in this predictive model. The features are ordered by their overall impact on the prediction, with the most influential at the top. Based on our previous inference regarding positive and more negative SHAP values, we can see that high features (shown on the plot in red) and low features (shown on the plot in blue) show different influences across the various SHAP values. For example, high values of “deposit_non_refund” and “previous_cancellations” seem to increase the risk of cancellation as shown by the high feature red on the positive side of the graph. While on the other hand,  low values such as “adr” (average daily rate), and “total_of_special_requests” exhibit a mix of both the positive and the negative impacts. This indicates that these features have influence on cancellation depending on specific other circumstances. Furthermore, “country”, the first feature on this summary plot seems to skew the model either way, being at the top indicates it has one of the highest impacts on the likelihood of cancellation. Its broad range of SHAP values also indicates variations in high or low cancellation rates depending on certain countries. This overall visualisation helps identify what factors most strongly affect cancellation risk, aiding in targeted decision-making. 
    \n **2) LIME:**
    \nLIME is an explainable AI tool that helps explain how a complex machine learning model makes its predictions for a specific instance. Unlike SHAP, which shows the impact of each feature across the entire dataset, LIME focuses on explaining a single prediction in detail. It does this by creating a simpler version of the model for that particular data point, like a basic linear model. 
In our project, we used LIME to understand why the model predicted that one specific booking would be cancelled. This helped us identify which features were most important for that prediction, making the model's behaviour easier to understand for that individual case. The LIME visualisation showed two sets of features: those that supported the cancellation (in blue) and those that supported not cancelling (in orange).
    """)
    st.image('lime.png', caption='', use_column_width=True)

    st.write("""
    **LIME Analysis and Interpretations:**
    \nThe LIME chart explains why our model predicted that a specific hotel booking had an 85% chance of being cancelled and a 15% chance of not being cancelled. The chart splits the features into two categories: those that support cancellation (shown in blue) and those that reduce the chance of cancellation (shown in orange). Features like previous cancellations and non-refundable deposits had the biggest impact on predicting cancellation, each contributing 0.56. This means that customers with a history of cancelling bookings or opting for non-refundable deposits were much more likely to cancel. For instance, a previous cancellations value of -0.10 suggests that the customer had some cancellations in the past, but not many. This small history of cancellations influenced the model towards predicting that they might cancel again, but it wasn't a strong push. 
On the other hand, features like required car parking spaces (contribution of 0.11), room type changes (0.10), and previous successful bookings (0.10) helped reduce the chance of cancellation, suggesting these customers were more committed. Specifically, required car parking spaces had a positive influence on the likelihood of not cancelling because it suggested a higher level of commitment to the travel plans. The value -0.26 for parking spaces means that fewer parking spaces were requested, but since any parking requirement indicates planning and commitment, it still contributed to reducing the cancellation likelihood, though only slightly. Additionally, features like room type changes and successful previous bookings indicated a level of flexibility and a history of reliability, both of which made the customer less likely to cancel.
The actual values for these features provided important context to understand why they influenced the prediction in a particular way. Overall, the LIME visualisation helped us identify the features that had the most influence on this prediction, confirmed that the model's decisions made sense, and added trust and transparency to our results.
    \n**Mutual Information (MI) Feature Selection:**
    \nWe used Mutual Information (MI) to identify the features most strongly linked to the target variable, allowing us to retain only those with the highest predictive power. MI was chosen because it captures both linear and non-linear relationships between features and the target, making it more insightful than metrics that focus solely on linear correlations. Since our dataset included a mix of categorical and numerical features, MI's ability to handle different data types made it an ideal choice for understanding the factors contributing to booking cancellations in a complex, real-world scenario.
By applying MI feature selection, we calculated how much each feature was related to our target variable, "is_cancelled" (whether the booking was cancelled or not). We then ranked the features based on their MI scores to determine which were the most informative and had the highest potential for accurate predictions. Finally, we visualized the top features using a bar plot, which allowed us to clearly compare their mutual information scores and understand their relative importance.
    """)
    st.image('mi.png', caption='', use_column_width=True)
    st.write("""
    **MI Feature Selection Analysis and Interpretations:**
    \nThis chart highlights the Top 10 Features most strongly related to the prediction outcome based on their Mutual Information (MI) scores. The higher the MI score, the more informative the feature is in predicting cancellations. deposit_No Deposit had the highest MI score (≈ 0.14), suggesting that bookings without a deposit were more likely to be cancelled, as customers lacked financial commitment. Similarly, deposit_Non Refund (≈ 0.13) was also associated with cancellations, which was unexpected; despite the financial penalty, many customers still cancelled, indicating a need for further investigation. lead_time (≈ 0.08) also showed that longer periods between booking and arrival increased the likelihood of cancellations, as customers had more time to reconsider. adr (Average Daily Rate) (≈ 0.08) suggested that higher booking costs could lead to more cancellations, possibly because customers found cheaper alternatives. The agent used to make the booking (≈ 0.07) and the customer's country of origin (≈ 0.07) both had notable impacts, possibly reflecting differences in cancellation policies or cultural and economic factors. total_of_special_requests (≈ 0.06) indicated that customers who made specific requests were more likely to follow through with their bookings, showing higher commitment. Meanwhile, previous_cancellations (≈ 0.06) indicated that customers with a history of cancellations were more likely to cancel again, highlighting the predictive value of past behaviour. room_type_changed (≈ 0.05) suggested that changes to the room type could indicate uncertainty, leading to a higher likelihood of cancellation. Lastly, required_car_parking_spaces (≈ 0.05) suggested that requesting parking was a sign of commitment, reducing the chance of cancellation. Overall, the Mutual Information scores helped us identify the features most predictive of cancellations, with factors like deposit type and lead time emphasising the importance of financial commitment and booking timing in predicting customer behaviour.
    """)
    st.write("""
    **Next Steps**
    \nWe were initially surprised by the model's result that non-refundable deposits were strongly associated with an increased likelihood of cancellations. Typically, we would expect that customers opting for non-refundable bookings would be more committed, as they have already invested money that they cannot get back, which should ideally reduce the cancellation rate. However, the model showed the opposite effect, indicating that customers with non-refundable deposits were more likely to cancel. This unexpected finding highlights the need for further investigation to understand why this might be happening. In our next steps, we plan to dive deeper into the data to explore possible reasons behind this trend, aiming to clarify this behaviour and adjust our model accordingly.

To enhance our final customer churn prediction, our team plans to extend our analysis by implementing other additional supervised learning algorithms such as KNN (K-Nearest Neighbors) and Multilayer Perceptrons (MLP) to name a few. These models will provide a diverse set of perspectives on the data, allowing us to compare performance metrics across various algorithmic approaches. Through training such models and algorithms alongside the existing XGBoost algorithm, we can assess what methods lead to the most accurate and generalizable predictions.

As mentioned earlier, the current XGBoost model is a basic wireframe with default parameters that can be made more sophisticated as a part of our next steps. We aim to implement hyperparameter optimization to maximise its predictive power. Some methods of hyperparameter optimization we are looking at implementing include grid search/randomised search which will aid us in fine tuning parameters such as learning rate, max depth, etc. This will overall aid in reducing overfitting and improving real model accuracy. The optimizations are catered to tailoring the model to the specific needs of our dataset and will work towards making use of XAI predictions to back our modifications.

Additionally, we plan on creating at least five new visualisations to deepen our understanding of this model’s behaviours and furthermore, communicate final results with more effective portrayals. This may include looking at more plots that compare model performance, confusion matrices for insights into our classification strategies, and enhanced explainable AI metrics for further interpretability. Overall, these next steps will contribute to a better tailored, and effective predictive model for hotel churn management.

    """)

    subheading("""Midterm Updated Gantt Chart:""")
    st.image('gantt_chart_midterm.png', caption='', use_column_width=True)

def display_section_5():
    st.markdown("<div id='section_5'></div>", unsafe_allow_html=True)
    st.header("References")
    st.markdown("""
    1. Antonio, N., Almeida, A. M., & Nunes, L. (2017). Predicting hotel booking cancellations to decrease uncertainty and increase revenue. *Tourism & Management Studies, 13*(2), 25–39. [https://doi.org/10.18089/tms.2017.13203](https://doi.org/10.18089/tms.2017.13203)
    
    2. Lin, Y. (2023). Research on the influencing factors of cancellation of hotel reservations. *Highlights in Science, Engineering and Technology, IFMPT 2023, 61*, 107–118. [https://doi.org/10.18089/rs.IFMPT2023-107118](https://doi.org/10.18089/rs.IFMPT2023-107118)
    
    3. Sánchez, E. C., Sánchez-Medina, A. J., & Pellejero, M. (2020). Identifying critical hotel cancellations using artificial intelligence. *Tourism Management Perspectives, 35*, Article 100718. [https://doi.org/10.1016/j.tmp.2020.100718](https://doi.org/10.1016/j.tmp.2020.100718)
    
    4. Özkurt, C. (2024). Transparency in decision-making: The role of explainable AI (XAI) in customer churn analysis. *Sakarya University of Applied Sciences*. [https://doi.org/10.21203/rs.3.rs-3937355/v1](https://doi.org/10.21203/rs.3.rs-3937355/v1)
    """)
    data = {
    "Name": ["Veer Kejriwal", "Shubhangi Asthana", "Stuti Singhal", "Raima Nawal", "Parul Methi"],
    "Contribution": [
        "Streamlit Setup, Methods Section",
        "Presentation Design",
        "Streamlit Design, Literature Review",
        "Gantt Chart, Dataset & Problem Definition",
        "Results, Expectations, Video Recording"
    ]
}
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)



    # Display the table
    subheading("""Contribution Table""")
    st.table(df)


    subheading("""Midterm Contribution Table""")

    data2 = {
    "Name": ["Veer Kejriwal", "Shubhangi Asthana", "Stuti Singhal", "Raima Nawal", "Parul Methi"],
    "Contribution": [
        "Implementing the machine learning algorithm, adding quantitative metrics, updated the github repo",
        "Completed the data visualisation and added the explainable AI metrics",
        "Completed the data visualisation and added explainable AI metrics",
        "Implemented code for the data preprocessing",
        "Wrote the midterm checkpoint writeup analyzing results, updated GANTT chart and contribution table"
    ]
}
    # Convert data to a pandas DataFrame
    df2 = pd.DataFrame(data2)
    st.table(df2)

# App layout
st.title("Hotel Reservation Cancellations: Predicting Customer Churn with Explainable AI")



# st.sidebar.markdown("""
# <style>
# /* Base style for the link */
# a.nav-button:link,
# a.nav-button:visited {
#     display: block;
#     width: 100%;
#     padding: 0.5em;
#     margin: 0.5em 0;
#     background-color: #FFFFFF;  /* White background */
#     color: #000000;             /* Black text */
#     text-align: center;
#     text-decoration: none;      /* Remove underline */
#     border-radius: 4px;
#     font-weight: bold;
# }

# /* Style for the link when hovered over or active */
# a.nav-button:hover,
# a.nav-button:active {
#     background-color: #D3D3D3;  /* Light grey background on hover */
#     color: #000000;             /* Ensure text remains black */
#     text-decoration: none;      /* Remove underline */
# }
# </style>
# """, unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
/* Base style for the link */
a.nav-button:link,
a.nav-button:visited {
    display: block;
    width: 100%;
    padding: 0.5em;
    margin: 0.5em 0;
    background-color: #FFFFFF;      /* White background */
    color: #489788;                 /* Text color */
    text-align: center;
    text-decoration: none;          /* Remove underline */
    # border: 2px solid #489788;      /* Border color */
    # border-radius: 4px;
    font-weight: bold;
}

/* Style for the link when hovered over or active */
a.nav-button:hover,
a.nav-button:active {
    background-color: #D3D3D3;      /* Light grey background on hover */
    color: #489788;                 /* Keep text color on hover */
    text-decoration: none;          /* Remove underline */
}
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown('<a href="#section_1" class="nav-button">Introduction</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section_2" class="nav-button">Dataset Description</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section_3" class="nav-button">Methods</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section_4" class="nav-button">Results</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#section_5" class="nav-button">References</a>', unsafe_allow_html=True)


# Display the sections
display_section_1()
display_section_2()
display_section_3()
display_section_4()
display_section_6()
display_section_5()

