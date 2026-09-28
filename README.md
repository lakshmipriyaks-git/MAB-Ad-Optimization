🎯 Multi-Armed Bandit (MAB) for Online Advertising

An interactive web application built with Python and Streamlit that demonstrates how **Multi-Armed Bandit (Thompson Sampling)** optimization dynamically allocates ad budget in real-time, minimizes opportunity cost (regret), and outperforms traditional fixed-split A/B testing.

🚀 Key Features
* **Exploration vs. Exploitation:** Automatically balances testing new ad variants with routing traffic to top performers.
* **Thompson Sampling Algorithm:** Uses Bayesian probability (Beta distribution) to learn click-through rates dynamically.
* **Cumulative Regret Analysis:** Visualizes opportunity cost savings over thousands of simulated impressions.
* **Interactive Localhost Dashboard:** Built with Streamlit so users can tweak CTR values and test parameters live in the browser.

🛠️ Techniques Used
1. **Exploratory Data Analysis (EDA):** Baseline inspection of performance variance.
2. **Multi-Armed Bandit Framework:** Algorithmic decision-making under uncertainty.
3. **Cumulative Reward & Regret Evaluation:** Measuring efficiency and convergence speed.
4. **Bayesian Modeling:** Using Beta distributions to track success probabilities.
