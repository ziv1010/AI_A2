## Churn Prediction Report

**Date:** 2025-12-06

### Top 5 High‑Risk Customers (Most Likely to Churn)
| Rank | Customer ID | Name | Churn Probability | Age | Balance | Products | Active Member | Complained | Geography | Tenure (years) | Credit Score |
|------|-------------|------|-------------------|-----|---------|----------|---------------|------------|-----------|----------------|--------------|
| 1 | 15696061 | Brownless | 99.9% | 34 | $101,633 | 1 | No | Yes | Germany | 1 | 581 |
| 2 | 15647311 | Hill | 99.8% | 41 | $83,808 | 1 | Yes | Yes | Spain | 1 | 608 |
| 3 | 15586310 | Ting | 99.4% | 30 | $169,462 | 1 | No | Yes | France | 4 | 578 |
| 4 | 15789484 | Hammond | 98.1% | 36 | $169,831 | 2 | Yes | Yes | Germany | 6 | 751 |
| 5 | 15586914 | Nepean | 98.0% | 36 | $123,841 | 2 | No | Yes | France | 6 | 659 |

### Top 5 Low‑Risk Customers (Least Likely to Churn)
| Rank | Customer ID | Name | Churn Probability | Age | Balance | Products | Active Member | Complained | Geography | Tenure (years) | Credit Score |
|------|-------------|------|-------------------|-----|---------|----------|---------------|------------|-----------|----------------|--------------|
| 1 | 15799932 | Iweobiegbunam | 0.0% | 24 | $0 | 2 | Yes | No | France | 10 | 812 |
| 2 | 15782159 | Ndubuagha | 0.0% | 28 | $67,640 | 2 | Yes | No | France | 8 | 850 |
| 3 | 15724527 | Forbes | 0.0% | 34 | $0 | 2 | Yes | No | France | 9 | 825 |
| 4 | 15693683 | Yuille | 0.0% | 29 | $97,086 | 2 | Yes | No | Germany | 8 | 814 |
| 5 | 15743760 | Davidson | 0.0% | 31 | $131,997 | 2 | Yes | No | France | 6 | 850 |

### Key Factors Driving Churn (Logistic Regression Coefficient Importance)
| Rank | Feature | Contribution to Churn Prediction |
|------|---------|-----------------------------------|
| 1 | Complain | 66.76% |
| 2 | IsActiveMember | 13.85% |
| 3 | Gender | 9.11% |
| 4 | NumOfProducts | 3.90% |
| 5 | HasCrCard | 2.95% |
| 6 | Card Type | 1.46% |
| 7 | Geography | 0.96% |
| 8 | Satisfaction Score | 0.51% |
| 9 | Age | 0.25% |
|10 | Tenure | 0.21% |

### Recommendations to Reduce Churn
1. **Address Customer Complaints Promptly** – Since complaints account for ~67% of the churn signal, implement a rapid‑response support system, track complaint resolution time, and follow up with dissatisfied customers.
2. **Engage Inactive Members** – Inactive members (IsActiveMember = No) contribute ~14% to churn risk. Offer personalized re‑engagement campaigns, such as targeted promotions or loyalty rewards.
3. **Tailor Communication by Gender** – Gender influences churn (~9%). Analyze gender‑specific usage patterns and craft relevant product bundles or messaging.
4. **Cross‑Sell/Up‑Sell Strategically** – Customers with fewer products are slightly more likely to churn. Encourage eligible customers to adopt additional services that fit their profile.
5. **Credit Card Ownership & Card Type** – Promote the benefits of having a credit card and consider offering upgraded card types to increase perceived value.
6. **Geography‑Specific Initiatives** – Although a smaller factor, regional preferences matter. Deploy localized marketing and support resources.
7. **Improve Satisfaction Scores** – Even modest impact, higher satisfaction reduces churn. Conduct regular NPS surveys and act on feedback.
8. **Age & Tenure** – Younger and newer customers show marginally higher churn; provide onboarding programs and mentorship for newer accounts.

**Overall Action Plan:** Prioritize complaint resolution and active‑member engagement, then layer additional tactics based on the secondary factors above. Monitoring these high‑risk customers closely and intervening early can significantly lower churn rates.
