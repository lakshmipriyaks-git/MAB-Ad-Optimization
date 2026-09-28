import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="MAB Ad Optimization", layout="wide")

st.title("🎯 Multi-Armed Bandit (MAB) for Online Advertising")
st.markdown("This interactive app demonstrates how a **Thompson Sampling** Multi-Armed Bandit dynamically optimizes ad spend in real-time compared to traditional approaches.")

# Sidebar controls for simulation parameters
st.sidebar.header("Simulation Settings")
n_steps = st.sidebar.slider("Number of Impressions", min_value=500, max_value=10000, value=2000, step=500)

st.sidebar.subheader("True CTRs of Ad Variants")
ctr_ad_0 = st.sidebar.slider("Ad A CTR", 0.01, 0.30, 0.05, 0.01)
ctr_ad_1 = st.sidebar.slider("Ad B CTR", 0.01, 0.30, 0.08, 0.01)
ctr_ad_2 = st.sidebar.slider("Ad C CTR", 0.01, 0.30, 0.15, 0.01)
ctr_ad_3 = st.sidebar.slider("Ad D CTR", 0.01, 0.30, 0.04, 0.01)

true_ctrs = [ctr_ad_0, ctr_ad_1, ctr_ad_2, ctr_ad_3]

# Simulator & Bandit classes
class AdSimulator:
    def __init__(self, true_ctr_rates):
        self.true_ctr_rates = true_ctr_rates
        
    def pull_arm(self, arm_index):
        return np.random.binomial(1, self.true_ctr_rates[arm_index])

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        
    def select_arm(self):
        samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.n_arms)]
        return np.argmax(samples)
    
    def update(self, arm_index, reward):
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1

# Run Simulation Button
if st.button("Run MAB Simulation"):
    simulator = AdSimulator(true_ctrs)
    bandit = ThompsonSamplingBandit(n_arms=len(true_ctrs))

    choices = []
    cumulative_regret = []
    optimal_ctr = max(true_ctrs)
    total_regret = 0

    for t in range(1, n_steps + 1):
        chosen_arm = bandit.select_arm()
        reward = simulator.pull_arm(chosen_arm)
        bandit.update(chosen_arm, reward)
        
        regret = optimal_ctr - true_ctrs[chosen_arm]
        total_regret += regret
        
        choices.append(chosen_arm)
        cumulative_regret.append(total_regret)

    # Layout results into columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Cumulative Regret Over Time")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cumulative_regret, color='#6c5ce7', linewidth=2)
        ax.set_xlabel('Impressions')
        ax.set_ylabel('Cumulative Regret')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Traffic Distribution Across Ads")
        choice_counts = pd.Series(choices).value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar([f"Ad {chr(65+i)}" for i in range(len(true_ctrs))], choice_counts.values, color='#00b894')
        ax2.set_xlabel('Ad Variants')
        ax2.set_ylabel('Total Impressions Received')
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig2)

    st.success("Simulation completed successfully! Notice how the traffic shifts heavily toward the ad with the highest true CTR.")