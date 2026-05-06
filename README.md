# The Silent Goodbye: "Predicting customer exits before it happens."
# Overview
The telecom industry is characterized by customer acquisition, retention, and the ever-evolving field of innovations that requires companies to predict customer behaviors and develop response mechanism. Customer retention is imperative as acquiring new customers is expensive than retaining, building brand loyalty is therefore key driver for telecom companies. 

By understanding why customers opt out of the industry can reduce exits or at least predict when they will exit and develop measures to retain them. This study focuses on building a machine learning classifier to predict whether a customer will likely exit SyriaTel in the near future. By identifying at-risk or dissatisfaction drivers’ customers early, SyriaTel can take proactive measures to improve customer satisfaction, develop targeted approaches and reduce exits and eventual revenue loss.
# Business Problem
SyriaTel is experiencing unprecedented customer attrition, where a significant portion of its subscribers exit their network over time. This exist have cascading effect on revenue loss, increased marketing cost and acquisition costs and eventually lack of value for money for the customers. The exits are silent, and reflect overtime and most telecommunication comes notice when it’s too late. This data analysis project seeks to solve that problem by developing a data driven solution that predicts exits before they happen and allow SyriaTel to develop appropriate intervention measures. 
# Business Understanding
In the telecom sector, customer exits are influenced by multiple factors, including:

- Service quality (e.g., dropped calls, network issues) 
- Pricing and billing transparency 
- Customer support experience 
- Usage behavior (declining activity can signal disengagement) 

From a telecom business perspective, not all customers are equally valuable. Customer retention is important and that what keeps the business running and profitable. Therefore, any model developed should not only predict exits but also help prioritize retention efforts. Understanding these dynamics allows us to translate raw data into meaningful business insights and align the model with real-world decision-making.
## Key Business Questions
This analysis is guided by practical, decision-oriented questions:

- Which customer demographic is most likely to exit/opt-out soon? 
- What are the driving factors that are most strongly associated with exits? 
- Can SyriaTel identify early warning signals/indicators before a customer exists? 
- How accurately can SyriaTel predict exits using historical data? 
- What actionable insights can SyriaTel deploy to retain high-risk customers? 
## Key Objectives of the Study / Data Analysis
This study aims to achieve the following:

- Create a reliable classification model to predict whether a customer will exit. 
- Evaluate the model performance using appropriate metrics on their ability to predict. 
- Identify key drivers of exits and insights fueling customers exits 
- Provide actionable insights that SyriaTel implement. 
- Propose retention strategies guided by data for high-risk customers
- ## Data set dictionary
- - **State**: Location of the customer                 
- **account length**: period of time he/she has been a customer 
- **area code**: Location code 
- **phone number**: phone number  
- **international plan**: International plan
- **voice mail plan**: Voice mail
- **number vmail messages**: total number of voice mail recorded
- **total day minutes**: day time usage 
- **total day calls**: day number of calls
- **total day charge**: amounts used per day
- **total eve minutes**: total evening time usage
- **total eve calls**: total calls made in the evening
- **total eve charge**: total amount of calls made in the evening
- **total night minutes**; total night time usage
- **total night calls**; total night calls
- **total night charge**; total amount of night calls
- **total intl minutes**; total int time usage
- **total intl calls**; total number of int calls
- **total intl charge**; total amount of int calls
- **customer service calls**; total number of calls to customer service 
- **churn**: exits
- # Exploration Data Analysis
- ## Loading dataset with necessary libraries.
- Loading libraries
- Loading dataset 'syria_tel.csv'
- # Data Understanding
- telecom_syria.info()
- telecom_syria.describe()
- telecom_syria.head()
- telecom_syria.isna().sum().sort_values(ascending=False)
- Insights from the dataset.

1. The data above shows 3,333 entries (rows) and 21 columns..
2. Although churn is a categorical (boolean) variable, it will be encoded as a binary numerical variable (0 and 1) to make it compatible with machine learning algorithms.
3. Some columns will be dropped as they are not necessary such as (phone number, state, charges and area code).
4. Some data requires feature engineering, which will be expounded later.
5. There is no missing data in all the columns and rows in our data set.
## Data Cleaning
Dropping columns
Creating customer usage patterns and behaviours.
Creating a new column for account length to check on loyalty.
Checking data telecom_syria.head()
Checking data imbalance 
Insights from the imbalance check.

1. This data is significantly imbalanced with 85.5% non-exits and 14.5% exits.
2. This imbalance can bias our models towards predicting the majority class, affecting the detection of actual exits.
4. We fix this with recall and F1-score rather than focusing on accuracy.
Checking the numerical values for the outlier effect.
International Plans comparison with Exits.
Day usage patterns vs Exits.
Total usage patterns and exits
Correlation heat map of target vs features
Voicemail vs exits
# EDA Summary 
- The data exploratory analysis reveals that customer exits are directly correlated with service dissatisfaction (measured through customer service calls), usage behavior, and service plan characteristics. 

- The dataset is imbalanced, necessitating appropriate handling during modeling. 

- Additionally, correlations among usage and charge variables highlight the need for feature selection to avoid redundancy.
- # Data Modelling.
- ## Selecting the Target (y) and features (X)
- ## Data Splitting, Train-Test Split.
- ## Using SMOTE on Data Imbalance.
- ## Data Scaling.
- ## Logistic Regression Modelling.
- ## Decision Tree Modelling.
- ## Hyperparameter Tuning.
- # Model Evaluation.
- ## Confusion Matrix
- Confusion Matrix Insights. 

1. Our model correctly predicted 533 loyal customers would stay, it managed to correctly predict 71 customers would exit and hence caught them.
2. Our Model wrongly caught 37 loyal customers, these ones aren't exiting, while our model didn't catch 26 cusotmers. 
3. Revenue protection: The model captures 73% of actual customer exits (71 out of 97), meaning the financial benefit of stopping these departures far outweighs the minor cost of the 37 false alarms.
4. Focus on efficiency: Because 66% of flagged customers (71 out of 108) are truly at risk, SyriaTel avoids "margin erosion" by not wasting unnecessary discounts on the 37 loyal customers who were incorrectly flagged.
5. Targeted intervention: For every 108 customers the model identifies, 71 are confirmed leavers, allowing management to move away from broad, expensive promotions toward a highly surgical, data-driven outreach.
6. ## Models Comparison.
7. Insights from Model Comparison 

1. The Decision Tree (Tuned) achieves the highest F1-Score of 0.70, indicating it provides the best overall balance between catching customer exits and maintaining accuracy.

2. With a Recall of 0.73, the model successfully identifies 73% of all actual customer exits, allowing SyriaTel to proactively intervene and protect revenue.

3. The model boasts the highest Precision of 0.68, meaning that 68% of customers flagged for retention are truly at risk of leaving.

4. By maintaining high precision, the model minimizes "marketing waste," ensuring that resources are not heavily spent on the 32% of loyal customers who were incorrectly flagged.

5. While Logistic Regression has a slightly higher recall (0.76), its extremely low precision (0.38) would result in a massive waste of resources compared to the surgical accuracy of the Tuned Decision Tree.
## Identifying the Drivers of Customer Exits.
Key insights from feature importance visualization

1. Key reasons driving customers to exit SyriaTel are customer service at 24%, international plan 20%, and total minutes 16% total charges at 18%
2. Important to classify the type or reasons for customer service calls.
# Final insights.
1. There "Frustration threshold": There is a tipping point where the number of calls correlates with exit, suggesting that if not resolved, it is a primary driver for customers exits. Customer loyalty drops off after the 3rd support call. The exit rate jumps from ~10% at three calls to 45.78% at four calls, eventually exceeding 60% for customers making five or more calls.

2. International plans customers show a higher rate of exits than those without, suggesting pricing or service quality issues. 

3. There is high vulnerability of high data users, especially during daytime hours and charges, being most profitable, their retention is imperative to check revenue loss. While high usage typically signals value, for SyriaTel it is a risk indicator. Customers with over 250 total day minutes have an exit rate of 46.95%, compared to only 10.95% for normal users.

4. Predictive Power: Using the Decision Tree (Tuned) model, we can now capture 73% of all actual churners before they leave, with a 66% accuracy rate on our intervention triggers..
# Recommendations
1. High-value customers retention: A strategic retention strategy that targets customers' usage patterns like improved plans, charges, and minutes. 
2. Improve customer experience, for example, fast resolutions of customer complaints, and monitor customers with repeated complaints. 
3. Support desk "Red Flag" system to spot customers' spikes on the 3rd calls and escalate the issue for further action. 
4. Package restructuring: Optimize on price and minutes to offer better packages like international plans or daytime offers. 
5. Customer acquisition should be prioritized, review the onboarding experience, and improve new customer incentives for a period. 
6. Targeted marketing focusing on high-risk customers and loyal customers. 
7. Integrate daily/weekly exit scoring and link it to customer experience teams to follow up.
# Conclusion
This analysis has demonstrated that a machine learning model is capable of predicting customer exits with strong performance. The tuned decision tree model provided the best balance between identifying at-risk customers and minimizing false alarms.

The analysis shows that customer exits are largely influenced by service experience, usage behavior, and customer tenure rather than simple demographic factors. Importantly, the model demonstrates that exits are predictable and can be proactively managed and retention efforts prioritized.

By leveraging these insights, SyriaTel can transition from a reactive to a proactive customer retention strategy, significantly reducing customer attrition and associated revenue loss.”
