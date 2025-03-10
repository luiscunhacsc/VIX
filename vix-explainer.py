import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import datetime

#######################################
# 1) Define callback functions:
#    - One to reset defaults
#    - One to set Lab parameters
#######################################
def reset_parameters():
    st.session_state["recent_vol_slider"] = 12.0
    st.session_state["vix_slider"] = 16.0
    st.session_state["mean_rev_speed_slider"] = 0.25
    st.session_state["mean_rev_level_slider"] = 16.0
    st.session_state["premium_factor_slider"] = 3.5
    st.session_state["prediction_days_slider"] = 30

def set_low_vol_parameters():
    st.session_state["recent_vol_slider"] = 8.0
    st.session_state["vix_slider"] = 11.0
    st.session_state["mean_rev_speed_slider"] = 0.25
    st.session_state["mean_rev_level_slider"] = 16.0
    st.session_state["premium_factor_slider"] = 3.5
    st.session_state["prediction_days_slider"] = 30

def set_high_vol_parameters():
    st.session_state["recent_vol_slider"] = 25.0
    st.session_state["vix_slider"] = 30.0
    st.session_state["mean_rev_speed_slider"] = 0.25
    st.session_state["mean_rev_level_slider"] = 16.0
    st.session_state["premium_factor_slider"] = 3.5
    st.session_state["prediction_days_slider"] = 30

def set_fear_parameters():
    st.session_state["recent_vol_slider"] = 15.0
    st.session_state["vix_slider"] = 28.0
    st.session_state["mean_rev_speed_slider"] = 0.25
    st.session_state["mean_rev_level_slider"] = 16.0
    st.session_state["premium_factor_slider"] = 3.5
    st.session_state["prediction_days_slider"] = 30

def set_complacency_parameters():
    st.session_state["recent_vol_slider"] = 15.0
    st.session_state["vix_slider"] = 10.0
    st.session_state["mean_rev_speed_slider"] = 0.25
    st.session_state["mean_rev_level_slider"] = 16.0
    st.session_state["premium_factor_slider"] = 3.5
    st.session_state["prediction_days_slider"] = 30
#######################################

# VIX prediction functions
def calculate_mean_reversion_adjustment(recent_vol, mean_rev_level, mean_rev_speed):
    """Calculate the mean reversion component of future volatility change"""
    return (mean_rev_level - recent_vol) * mean_rev_speed

def calculate_expected_vix(recent_vol, mean_rev_adjustment, premium_factor):
    """Calculate what VIX 'should' be given current conditions"""
    # The paper uses a relationship between squared values, but we simplify here
    vol_adjustment = mean_rev_adjustment
    volatility_premium = premium_factor
    
    return recent_vol + vol_adjustment + volatility_premium

def predict_future_volatility(recent_vol, vix, expected_vix, mean_rev_adjustment):
    """Predict future volatility based on VIX and recent volatility"""
    # Mean reversion component
    mean_rev_component = mean_rev_adjustment
    
    # VIX deviation component (difference between actual VIX and expected VIX)
    vix_deviation = vix - expected_vix
    
    # Predicted change in volatility
    predicted_change = mean_rev_component + (vix_deviation * 0.5)  # Dampening factor
    
    # Future volatility prediction
    future_vol = recent_vol + predicted_change
    
    return future_vol, predicted_change

def simulate_vix_path(current_vix, future_vol, days, mean_rev_level, mean_rev_speed, noise_level=0.15):
    """Simulate a potential path for VIX over future days"""
    vix_path = [current_vix]
    vol_path = [future_vol]
    
    for i in range(1, days):
        # Mean reversion for volatility
        vol_mr = calculate_mean_reversion_adjustment(vol_path[-1], mean_rev_level, mean_rev_speed)
        
        # Add some noise to volatility path
        vol_noise = np.random.normal(0, noise_level * vol_path[-1])
        new_vol = max(5, vol_path[-1] + vol_mr + vol_noise)
        vol_path.append(new_vol)
        
        # VIX follows volatility with a premium and some noise
        vix_premium = 3.5 + 0.2 * np.random.randn()
        vix_noise = np.random.normal(0, noise_level * vix_path[-1])
        new_vix = max(5, new_vol + vix_premium + vix_noise)
        vix_path.append(new_vix)
    
    return vix_path, vol_path

# Configure the Streamlit app
st.set_page_config(layout="wide", page_title="VIX Explainer")
st.title("📊 Understanding VIX: Market's Fear Gauge")
st.markdown("Analyze how VIX (Volatility Index) relates to market volatility and sentiment, and learn to interpret its signals.")

# Sidebar for input parameters
with st.sidebar:
    st.header("⚙️ Parameters")
    
    st.button("↺ Reset Parameters", on_click=reset_parameters)

    recent_vol = st.slider("Recent Realized Volatility (%)", 5.0, 50.0, 12.0, key='recent_vol_slider')
    vix = st.slider("Current VIX Level", 5.0, 50.0, 16.0, key='vix_slider')
    mean_rev_speed = st.slider("Mean Reversion Speed", 0.1, 0.5, 0.25, 0.05, key='mean_rev_speed_slider')
    mean_rev_level = st.slider("Mean Reversion Level (%)", 10.0, 25.0, 16.0, key='mean_rev_level_slider')
    premium_factor = st.slider("Volatility Premium", 1.0, 6.0, 3.5, 0.5, key='premium_factor_slider')
    prediction_days = st.slider("Forecast Horizon (days)", 10, 90, 30, 5, key='prediction_days_slider')

    # Disclaimer and license
    st.markdown("---")
    st.markdown(
    """
    **⚠️ Disclaimer**  
    *Educational purposes only. No accuracy guarantees. Do not use as investment advice.*  
    
    <small>
    The author does not engage in volatility trading and does not endorse it for non-professional investors. 
    All information provided is for educational purposes only and should not be construed as financial or 
    investment advice. Trading volatility involves significant risks and may not be suitable for all investors. 
    Always consult a qualified financial professional before making any investment decisions.
    </small>
    """,
    unsafe_allow_html=True
    )

    
    st.markdown("""
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">By Luís Simões da Cunha, 2025</span>
    </div>
    """, unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Tool", 
    "📚 Theory Behind VIX", 
    "📖 Comprehensive Tutorial", 
    "🛠️ Practical Labs",
    "🧮 Playground"
])

with tab1:
    # Calculate predictions
    mean_rev_adjustment = calculate_mean_reversion_adjustment(recent_vol, mean_rev_level, mean_rev_speed)
    expected_vix_value = calculate_expected_vix(recent_vol, mean_rev_adjustment, premium_factor)
    future_vol, predicted_change = predict_future_volatility(recent_vol, vix, expected_vix_value, mean_rev_adjustment)
    
    # Simulate a potential path for VIX
    vix_path, vol_path = simulate_vix_path(vix, future_vol, prediction_days, mean_rev_level, mean_rev_speed)
    
    # VIX status determination
    vix_state = "NORMAL"
    vix_deviation = vix - expected_vix_value
    
    if vix_deviation > 5:
        vix_state = "HIGH (Market Fear)"
    elif vix_deviation < -5:
        vix_state = "LOW (Market Complacency)"
    
    # Display results in columns
    col1, col2 = st.columns([1, 3])
    with col1:
        st.success(f"### VIX Status: **{vix_state}**")
        
        # VIX Interpretation
        st.markdown("### VIX Analysis")
        st.markdown(f"""
        - **Recent Volatility:** `{recent_vol:.1f}%`
        - **Current VIX:** `{vix:.1f}`
        - **Expected VIX:** `{expected_vix_value:.1f}`
        - **VIX Deviation:** `{vix_deviation:.1f}`
        - **Predicted Vol Change:** `{predicted_change:.1f}%`
        - **Future Volatility:** `{future_vol:.1f}%`
        """)
        
        # VIX Interpretation
        st.info(f"""
        ### Interpretation:
        
        The VIX is currently **{abs(vix_deviation):.1f}%** {'above' if vix_deviation > 0 else 'below'} its expected level.
        
        This suggests the market is {'displaying fear' if vix_deviation > 0 else 'displaying complacency'}. Based on historical patterns, we might expect volatility to {'increase' if predicted_change > 0 else 'decrease'} by approximately {abs(predicted_change):.1f}% over the next 30 days.
        
        **Key insight:** {
            "Markets appear more fearful than warranted by current conditions. Historically, this has preceded volatility increases, but also potential market bottoms." if vix_deviation > 5 else 
            "Markets appear more complacent than warranted by current conditions. Historically, this has preceded volatility spikes and potential market corrections." if vix_deviation < -5 else
            "Markets appear to be pricing volatility in line with historical norms relative to current conditions."
        }
        """)

    with col2:
        # Generate path simulation graph
        fig, ax = plt.subplots(figsize=(10, 5))
        days = list(range(prediction_days))
        
        ax.plot(days, vix_path, label='Projected VIX Path', color='darkorange', linewidth=2)
        ax.plot(days, vol_path, label='Projected Realized Volatility Path', color='darkblue', linewidth=2, alpha=0.7)
        
        # Add current points
        ax.scatter(0, vix_path[0], color='red', s=100, label='Current VIX')
        ax.scatter(0, vol_path[0], color='blue', s=100, label='Current Realized Vol')
        
        # Add expected VIX
        ax.axhline(y=expected_vix_value, linestyle='--', color='green', alpha=0.7, label='Expected VIX Level')
        
        ax.set_title(f"VIX and Volatility Projection for Next {prediction_days} Days", fontweight='bold')
        ax.set_xlabel("Days Forward")
        ax.set_ylabel("Volatility Level (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # VIX Prediction Confidence
        st.warning("""
        **📉 Projection Confidence**
        
        This simulation includes randomness to reflect market uncertainty. The path shown is just one of many possible outcomes. Longer horizons have lower confidence levels.
        
        As VIX measures expected volatility over the next 30 days, the prediction confidence is highest for the near term and decreases beyond that window.
        """)

with tab2:
    st.markdown("""
    ## The VIX Index: Mathematical Foundation
    
    ### What is VIX?
    
    The VIX (Volatility Index), often called the "fear gauge," is a real-time market index representing the market's expectations for volatility over the coming 30 days. It's calculated from S&P 500 index options and represents the market's forecast of future price fluctuations.
    
    ### Core Concepts
    
    **1. Implied vs. Realized Volatility**
    
    VIX represents **implied volatility** - the market's forecast of a likely movement in the S&P 500 Index. This differs from **realized (or historical) volatility**, which measures actual price fluctuations that have occurred in the past.
    
    **2. The Volatility Premium**
    
    VIX typically overestimates subsequent realized volatility by about 4-5 percentage points on average. This is called the **volatility premium** and exists because:
    
    - Investors pay for "insurance" against market declines
    - Option sellers demand compensation for volatility risk
    - Supply/demand imbalances in the options market
    
    **3. Mean Reversion in Volatility**
    
    Volatility is **mean-reverting** - it tends to return to a long-term average level after periods of extremes:
    
    $$
    \\text{Expected Change in Volatility} = \\kappa(\\theta - \\sigma_t)
    $$
    
    Where:
    - $\\kappa$ is the speed of mean reversion
    - $\\theta$ is the long-term average volatility
    - $\\sigma_t$ is the current volatility
    
    **4. VIX Calculation**
    
    VIX is calculated using S&P 500 option prices across multiple strike prices:
    
    $$
    \\text{VIX} = 100 \\times \\sqrt{\\frac{2e^{rT}}{T}\\sum_i \\frac{\\Delta K_i}{K_i^2}Q(K_i) - \\frac{1}{T}\\left[\\frac{F}{K_0}-1\\right]^2}
    $$
    
    Where:
    - $T$ is time to expiration
    - $F$ is the forward index level
    - $K_0$ is the first strike below $F$
    - $K_i$ is the strike price of the $i$-th out-of-money option
    - $\\Delta K_i$ is the interval between strike prices
    - $Q(K_i)$ is the midpoint of the bid-ask spread for each option
    - $r$ is the risk-free interest rate
    
    **5. Understanding VIX Levels**
    
    General interpretation of VIX levels:
    
    | VIX Level | Market State | Interpretation |
    |-----------|--------------|----------------|
    | Below 12  | Low Volatility | Market complacency, potential build-up of risks |
    | 12-20     | Normal Volatility | Healthy market functioning |
    | 20-30     | Elevated Volatility | Uncertainty, moderate stress |
    | Above 30  | High Volatility | Fear, crisis, significant market stress |
    
    ### Reading VIX Properly
    
    A key insight from research is that VIX should be evaluated relative to **current realized volatility**. The formula for "Expected VIX" is:
    
    $$
    \\text{Expected VIX} = \\text{Recent Volatility} + \\text{Mean Reversion Adjustment} + \\text{Volatility Premium}
    $$
    
    The difference between actual VIX and Expected VIX provides a measure of market sentiment that may predict future volatility changes.
    """)
    
    with st.expander("🔍 Hands-On Theoretical Exercise"):
        st.markdown("""
        **Calculate Expected VIX:**
        
        1. Start with Recent Volatility = 12%
        2. Mean Reversion Level = 16%
        3. Mean Reversion Speed = 0.25
        4. Volatility Premium = 3.5%
        
        Mean Reversion Adjustment = (16% - 12%) × 0.25 = 1%
        
        Expected VIX = 12% + 1% + 3.5% = 16.5%
        
        If actual VIX = 25%, then VIX Deviation = 25% - 16.5% = 8.5%
        
        This indicates significant market fear that may predict:
        1. Higher future volatility
        2. A potential contrarian buying opportunity
        """)

with tab3:
    st.markdown("""
    ## Welcome to the VIX Learning Tool!
    
    **What this tool does:**  
    This interactive calculator helps you visualize how VIX levels relate to market volatility and sentiment, and what they might be telling us about future market conditions.
    
    ### Quick Start Guide
    
    1. **Adjust Parameters** (Left Sidebar):
       - Set recent realized volatility
       - Set current VIX level
       - Adjust mean reversion parameters
       - Set the forecast horizon
    
    2. **View Results** (Main Panel):
       - VIX status and deviation from expected level
       - Projected volatility path
       - Market sentiment interpretation
    
    3. **Try These Examples**:
       - 🎚️ Click "Set Low Vol Environment" to see a typical low-volatility market
       - ⚡ Click "Set High Vol Environment" to see a typical crisis scenario
       - 😨 Click "Set Fear Scenario" to see what happens when VIX is elevated relative to realized vol
       - 😴 Click "Set Complacency Scenario" to see what happens when VIX is low relative to realized vol
       - 🔄 Try adjusting the Mean Reversion Speed to see how quickly volatility normalizes
       - 📊 Experiment with the Volatility Premium to see how it affects VIX predictions
    
    ### Key Features to Explore
    - **VIX Deviation**: The difference between actual and expected VIX, indicating market sentiment
    - **Volatility Projections**: See how volatility might evolve based on current conditions
    - **Scenario Testing**: Evaluate what different market environments mean for future volatility
    
    **Pro Tip:** Use the parameter sliders to create your own scenarios and test their implications!
    """)

    # Button row for scenario presets
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Set Low Vol Environment", on_click=set_low_vol_parameters)
    with col2:
        st.button("Set High Vol Environment", on_click=set_high_vol_parameters)
    with col3:
        st.button("Set Fear Scenario", on_click=set_fear_parameters)
    with col4:
        st.button("Set Complacency Scenario", on_click=set_complacency_parameters)

# Tab 4: Practical Labs
with tab4:
    st.header("🔬 Practical VIX Labs")
    st.markdown("""
    Welcome to the **Practical VIX Labs** section! Each lab provides a real-world scenario or demonstration 
    to help you apply VIX analysis in a hands-on way.
    
    Use the **"Set Lab Parameters"** buttons to jump directly to recommended settings for each scenario.
    Experiment, take notes, and enjoy exploring how VIX behaves under different market conditions!
    """)

    # --- Additional Disclaimer ---
    st.warning("""
    **Disclaimer**:  
    The author does *not* recommend volatility trading as an investment strategy, 
    nor does he recommend that any retail investor should engage in it. 
    This material is purely for educational and illustrative purposes.
    """)

    # A radio to choose one of the labs
    lab_choice = st.radio(
        "Select a lab to view:",
        ("Lab 1: VIX as a Contrarian Indicator",
         "Lab 2: VIX Spikes and Market Bottoms",
         "Lab 3: Low VIX and Market Complacency",
         "Lab 4: VIX Term Structure",
         "Lab 5: VIX and Market Returns"),
        index=0
    )

    # ---------------- Lab 1 ----------------
    if lab_choice == "Lab 1: VIX as a Contrarian Indicator":
        st.subheader("🔄 Lab 1: VIX as a Contrarian Indicator")
        st.markdown("""
        **Real-World Scenario:**  
        You're a portfolio manager considering whether to increase or decrease equity exposure. The S&P 500 has 
        fallen 7% over the past two weeks, and VIX has spiked to 28% while realized volatility is at 18%. 
        Is this an opportunity to buy the dip, or a warning sign of more volatility to come?

        ---
        **Beginner Explanation of What's Happening:**

        - **VIX Elevation**: When VIX is significantly higher than recent realized volatility and its expected level,
          it indicates heightened fear in the market.
        
        - **Contrarian Signal**: Extremely elevated VIX levels have historically been associated with market bottoms,
          as they represent "peak fear" which often occurs near the end of market selloffs.
          
        - **Mean Reversion**: Both volatility and market fear tend to mean-revert over time, suggesting that
          periods of extreme fear are typically followed by normalization.

        **Learning Objective:**  
        - Understand how to use VIX as a potential contrarian indicator
        - Identify when VIX may be signaling "peak fear" versus justified concern

        ---
        **Suggested Steps**:
        1. Click "**Set Fear Scenario**" button on the Tutorial tab.
        2. Observe that VIX is significantly above the expected VIX level.
        3. Note the VIX deviation of approximately +13 points.
        4. Examine the projected path, which suggests volatility will likely increase but then revert to mean.
        5. Consider that extremely high VIX levels often accompany market bottoms.

        **💡 Reflection Questions:**  
        - Why do extreme VIX spikes often coincide with good entry points for equities?
        - How could you use VIX deviation to gauge when fear is excessive versus justified?
        - What other factors should you consider besides VIX when making contrarian bets?
        """)

    # ---------------- Lab 2 ----------------
    elif lab_choice == "Lab 2: VIX Spikes and Market Bottoms":
        st.subheader("📉 Lab 2: VIX Spikes and Market Bottoms")
        st.markdown("""
        **Real-World Scenario:**  
        You're analyzing historical market crashes and recoveries. During major market events (like 2008, 2020), 
        VIX often spikes dramatically to levels above 40, sometimes reaching as high as 80. What can these 
        extreme readings tell us about market bottoming processes?

        **Learning Objective:**  
        - Examine how extreme VIX readings relate to market bottoms
        - Understand the concept of volatility clustering and its implications

        ---
        **Suggested Steps**:
        1. Go to the Tutorial tab and click "**Set High Vol Environment**".
        2. Adjust VIX to 40 or higher to simulate a crisis environment.
        3. Note how the projected path shows a dramatic fall in VIX over time (though still with high volatility).
        4. Consider that extreme VIX readings rarely persist for extended periods.
        5. Recognize that VIX spikes often occur near, but slightly before, ultimate market bottoms.

        **Key Insight**:  
        - VIX above 40 represents extreme fear that historically doesn't persist
        - The rate of change in VIX can be as important as its absolute level
        - Look for declining VIX even as markets make new lows - a potential positive divergence

        **💡 Reflection Questions:**  
        - Why might a VIX "higher high" accompanied by a market "lower low" be potentially bullish?
        - How does the concept of mean reversion apply differently during crisis periods?
        """)

    # ---------------- Lab 3 ----------------
    elif lab_choice == "Lab 3: Low VIX and Market Complacency":
        st.subheader("😴 Lab 3: Low VIX and Market Complacency")
        st.markdown("""
        **Real-World Scenario:**  
        Markets have been calm for an extended period, with VIX hovering around 10-12% while realized 
        volatility is at 14-15%. Does this unusually low VIX (relative to realized volatility) signal 
        complacency that might precede a market correction?

        **Learning Objective:**  
        - Investigate how unusually low VIX levels can signal market complacency
        - Understand the concept of the "volatility risk premium" and when it compresses

        ---
        **Suggested Steps**:
        1. Go to the Tutorial tab and click "**Set Complacency Scenario**".
        2. Note that VIX is below the expected level based on current realized volatility.
        3. Observe that the VIX deviation is negative, indicating potential complacency.
        4. Examine the projected path, which suggests volatility might increase from currently suppressed levels.
        5. Consider historical periods when low VIX preceded market corrections.

        **Why This Matters**:  
        - Unusually low VIX can indicate investor complacency about potential risks
        - When VIX is below expected levels, markets may be vulnerable to volatility shocks
        - The "volatility risk premium" tends to expand rapidly during market stress

        **💡 Reflection Questions:**  
        - Why might a persistently low VIX be a warning sign rather than confirmation of a healthy market?
        - What types of market events tend to cause the most rapid VIX spikes from low levels?
        """)

    # ---------------- Lab 4 ----------------
    elif lab_choice == "Lab 4: VIX Term Structure":
        st.subheader("📈 Lab 4: VIX Term Structure and Volatility Expectations")
        st.markdown("""
        **Real-World Scenario:**  
        You're examining not just the standard 30-day VIX, but the entire VIX term structure (VIX futures
        for different expirations). Currently, the term structure is inverted, with near-term VIX futures 
        trading higher than longer-dated futures. What does this tell us about market expectations?

        **Learning Objective:**  
        - Understand how the VIX term structure reflects market expectations across different time horizons
        - Learn to interpret contango versus backwardation in the VIX futures curve

        ---
        **Suggested Steps**:
        1. Go to the Tutorial tab and use the "**Set High Vol Environment**" or "**Set Fear Scenario**" buttons.
        2. Imagine that the current 30-day VIX is elevated, but 3-month and 6-month VIX futures are progressively lower.
        3. This inverted term structure (backwardation) suggests the market expects current volatility to be temporary.
        4. Consider that VIX term structure inversion often occurs during market stress but tends to normalize.

        **Takeaway**:  
        - An inverted VIX term structure (backwardation) typically occurs during market stress
        - Contango (upward sloping term structure) is the normal state and suggests market calm
        - The steepness of the curve in either direction provides information about expected volatility changes

        **💡 Reflection Questions:**  
        - Why does the VIX term structure typically slope upward during normal market conditions?
        - How might changes in the VIX term structure provide early warning of market regime changes?
        """)

    # ---------------- Lab 5 ----------------
    else:  # lab_choice == "Lab 5: VIX and Market Returns"
        st.subheader("💰 Lab 5: VIX and Market Returns")
        st.markdown("""
        **Real-World Scenario:**  
        You're studying the relationship between VIX levels/changes and subsequent market returns. 
        Historical data suggests that extremely high VIX levels often precede strong market returns,
        while very low VIX readings might precede subpar returns. How can you use this information?

        **Learning Objective:**  
        - Examine the historical relationship between VIX and subsequent market returns
        - Understand how to incorporate VIX signals into a broader market analysis framework

        ---
        **Suggested Steps**:
        1. Compare scenarios using different VIX environments on the Tutorial tab.
        2. Note that very high VIX (> 30) has historically preceded above-average 12-month returns.
        3. Very low VIX (< 12) has historically preceded below-average 12-month returns.
        4. Consider that VIX reflects market-implied risk premium - when risk perception is high, 
           required returns are high, creating potential opportunities.

        **Use Case**:  
        - Extreme VIX readings can serve as a contrarian indicator for long-term investors
        - VIX trends and changes matter as much as absolute levels
        - Best used as one tool among many rather than in isolation

        **💡 Reflection Questions:**  
        - Why might extremely high VIX levels be associated with strong forward returns?
        - How would you incorporate VIX readings into a broader market analysis framework?
        - What market environments might cause this relationship to break down?
        """)

with tab5:
    st.header("🧮 Playground: Interactive VIX Learning")
    
    st.markdown("""
    Welcome to the VIX Playground! This section offers interactive activities to deepen your understanding 
    of VIX dynamics and volatility concepts through hands-on experimentation.
    """)
    
    activity = st.selectbox(
        "Choose an activity:",
        [
            "VIX Market Simulator",
            "Historical VIX Patterns",
            "VIX Prediction Challenge",
            "Volatility Regime Quiz"
        ]
    )
    
    if activity == "VIX Market Simulator":
        st.subheader("VIX Market Simulator")
        st.markdown("""
        This simulator lets you create different market scenarios and see how VIX might respond.
        
        **Instructions:**
        1. Configure market conditions
        2. Run the simulation
        3. Observe how VIX and realized volatility evolve
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Market Configuration")
            
            market_trend = st.radio("Market Trend", ["Bull Market", "Bear Market", "Sideways", "Crash"])
            vol_regime = st.radio("Volatility Regime", ["Low", "Normal", "High", "Extreme"])
            event_probability = st.slider("Event Probability (%)", 0, 100, 10)
            simulation_days = st.slider("Simulation Days", 30, 252, 60)
            
            run_simulation = st.button("Run Simulation")
        
        with col2:
            st.markdown("### Simulation Results")
            
            if run_simulation:
                # Set up simulation parameters based on selections
                if market_trend == "Bull Market":
                    base_vol = 10
                    drift = -0.1
                elif market_trend == "Bear Market":
                    base_vol = 20
                    drift = 0.1
                elif market_trend == "Sideways":
                    base_vol = 15
                    drift = 0
                else:  # Crash
                    base_vol = 35
                    drift = 0.3
                
                if vol_regime == "Low":
                    vol_multiplier = 0.7
                elif vol_regime == "Normal":
                    vol_multiplier = 1.0
                elif vol_regime == "High":
                    vol_multiplier = 1.5
                else:  # Extreme
                    vol_multiplier = 2.5
                
                # Generate simulation
                days = list(range(simulation_days))
                
                # Starting values
                start_vol = base_vol * vol_multiplier
                start_vix = start_vol + 4 + (2 * np.random.randn())
                
                # Create paths with randomness
                vol_path = [start_vol]
                vix_path = [start_vix]
                
                for i in range(1, simulation_days):
                    # Check for random events
                    event = np.random.rand() < (event_probability / 100)
                    event_multiplier = 1.5 if event else 1.0
                    
                    # Update volatility with mean reversion, drift, and randomness
                    vol_noise = np.random.normal(0, 0.1 * vol_path[-1])
                    mean_rev = 0.05 * (15 - vol_path[-1])
                    new_vol = max(5, vol_path[-1] + mean_rev + drift + vol_noise)
                    new_vol *= event_multiplier
                    vol_path.append(new_vol)
                    
                    # VIX follows volatility with a premium and some noise
                    vix_premium = 3.5 + 0.5 * np.random.randn()
                    vix_noise = np.random.normal(0, 0.15 * vix_path[-1])
                    new_vix = max(5, new_vol + vix_premium + vix_noise)
                    new_vix *= event_multiplier
                    vix_path.append(new_vix)
                
                # Plot the results
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(days, vix_path, label='VIX', color='darkorange', linewidth=2)
                ax.plot(days, vol_path, label='Realized Volatility', color='darkblue', linewidth=2, alpha=0.7)
                
                ax.set_title(f"{market_trend} with {vol_regime} Volatility Simulation", fontweight='bold')
                ax.set_xlabel("Days")
                ax.set_ylabel("Volatility Level (%)")
                ax.grid(alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Key statistics
                avg_vix = np.mean(vix_path)
                avg_vol = np.mean(vol_path)
                avg_premium = avg_vix - avg_vol
                max_vix = np.max(vix_path)
                min_vix = np.min(vix_path)
                
                st.markdown(f"""
                **Simulation Statistics:**
                - Average VIX: `{avg_vix:.2f}%`
                - Average Vol: `{avg_vol:.2f}%`
                - Average Premium: `{avg_premium:.2f}%`
                - Max VIX: `{max_vix:.2f}%`
                - Min VIX: `{min_vix:.2f}%`
                - VIX Range: `{max_vix - min_vix:.2f}%`
                """)
                
                st.info(f"""
                **Key Insights:**
                
                In this {market_trend.lower()} scenario with {vol_regime.lower()} volatility:
                
                {
                    "VIX maintains a relatively low level with small volatility premium. Occasional spikes may still occur with the specified event probability." if market_trend == "Bull Market" else
                    "VIX shows elevated levels with a higher volatility premium due to market uncertainty and downside protection demand." if market_trend == "Bear Market" else
                    "VIX fluctuates around a moderate level with typical volatility premium. Directionless markets can sometimes create their own uncertainty." if market_trend == "Sideways" else
                    "VIX spikes dramatically, reflecting extreme fear. The volatility premium often expands significantly during crash scenarios as demand for protection surges."
                }
                """)
            
            else:
                st.info("Click 'Run Simulation' to see results")
    
    elif activity == "Historical VIX Patterns":
        st.subheader("Historical VIX Patterns")
        st.markdown("""
        Explore how VIX has behaved during significant market events throughout history.
        """)
        
        # Sample historical events
        events = {
            "2008 Financial Crisis": {
                "date": "October 2008",
                "pre_vix": 25,
                "peak_vix": 80,
                "post_vix": 40,
                "days_to_peak": 21,
                "days_to_normalize": 95,
                "description": "Global financial crisis triggered by the subprime mortgage collapse led to Lehman Brothers' bankruptcy and a market crash."
            },
            "2010 Flash Crash": {
                "date": "May 6, 2010",
                "pre_vix": 20,
                "peak_vix": 42,
                "post_vix": 25,
                "days_to_peak": 1,
                "days_to_normalize": 12,
                "description": "The Dow Jones dropped about 1,000 points (around 9%) within minutes, only to recover most losses by close."
            },
            "2020 COVID Crash": {
                "date": "March 2020",
                "pre_vix": 15,
                "peak_vix": 82,
                "post_vix": 30,
                "days_to_peak": 18,
                "days_to_normalize": 60,
                "description": "Global pandemic fears caused the fastest 30% market drop in history, with unprecedented volatility and liquidity challenges."
            },
            "2022 Rate Hike Fears": {
                "date": "Jan-Feb 2022",
                "pre_vix": 17,
                "peak_vix": 36,
                "post_vix": 28,
                "days_to_peak": 25,
                "days_to_normalize": 40,
                "description": "Markets reacted to signals of aggressive Fed rate hikes to combat inflation, particularly affecting growth stocks."
            }
        }
        
        selected_event = st.selectbox("Select a historical event:", list(events.keys()))
        event = events[selected_event]
        
        # Display event description
        st.markdown(f"""
        **{selected_event}** (*{event['date']}*)
        
        {event['description']}
        """)
        
        # Simulate the VIX pattern for the event
        days_before = 20
        days_after = event['days_to_normalize'] + 20
        
        # Make sure arrays will have matching lengths
        days = list(range(-days_before, event['days_to_peak'] + days_after))
        vix_values = []
        
        # Pre-event period
        for i in range(days_before):
            random_factor = 0.05 * event['pre_vix'] * np.random.randn()
            vix_values.append(event['pre_vix'] + random_factor)
        
        # Event buildup to peak
        for i in range(event['days_to_peak']):
            progress = (i + 1) / event['days_to_peak']
            current_vix = event['pre_vix'] + progress * (event['peak_vix'] - event['pre_vix'])
            random_factor = 0.07 * current_vix * np.random.randn()
            vix_values.append(current_vix + random_factor)
        
        # Event cool down period
        for i in range(days_after):
            progress = min(1, (i + 1) / event['days_to_normalize'])
            current_vix = event['peak_vix'] - progress * (event['peak_vix'] - event['post_vix'])
            random_factor = 0.1 * current_vix * np.random.randn()
            vix_values.append(current_vix + random_factor)
            
        # Ensure arrays have same length
        assert len(days) == len(vix_values), f"Length mismatch: days={len(days)}, vix_values={len(vix_values)}"
        
        # Plot the event
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Mark the event date and peak
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Event Start')
        ax.axvline(x=event['days_to_peak'], color='darkred', linestyle='--', alpha=0.7, label='VIX Peak')
        
        ax.plot(days, vix_values, color='darkorange', linewidth=2.5)
        
        # Annotations
        ax.annotate('Pre-Event', xy=(-days_before/2, event['pre_vix']), 
                   xytext=(-days_before/2, event['pre_vix'] + 10),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   ha='center')
        
        ax.annotate('Peak Fear', xy=(event['days_to_peak'], event['peak_vix']), 
                   xytext=(event['days_to_peak'] - 5, event['peak_vix'] + 15),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   ha='center')
        
        ax.annotate('Normalization', xy=(event['days_to_peak'] + event['days_to_normalize']/2, 
                                       (event['peak_vix'] + event['post_vix'])/2), 
                   xytext=(event['days_to_peak'] + event['days_to_normalize']/2 - 5, 
                          (event['peak_vix'] + event['post_vix'])/2 + 15),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   ha='center')
        
        ax.set_title(f"VIX Pattern During {selected_event}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Days Relative to Event Start")
        ax.set_ylabel("VIX Level")
        ax.grid(alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Key insights about this event
        st.markdown(f"""
        ### Key VIX Insights from this Event
        
        - **Pre-Event VIX:** {event['pre_vix']} - {"Low complacency" if event['pre_vix'] < 15 else "Normal volatility" if event['pre_vix'] < 25 else "Already elevated"}
        - **Peak VIX:** {event['peak_vix']} - {"Moderate stress" if event['peak_vix'] < 30 else "Significant fear" if event['peak_vix'] < 50 else "Extreme panic"}
        - **Days to Peak:** {event['days_to_peak']} - {"Sudden shock" if event['days_to_peak'] < 5 else "Rapid deterioration" if event['days_to_peak'] < 15 else "Gradual buildup"}
        - **Normalization Period:** {event['days_to_normalize']} days - {"Quick recovery" if event['days_to_normalize'] < 20 else "Normal recovery" if event['days_to_normalize'] < 45 else "Extended stress"}
        
        **VIX Behavior Pattern:**
        
        {
            "Sudden shock pattern with extremely rapid VIX spike and quick normalization. Classic 'flash crash' signature." if selected_event == "2010 Flash Crash" else
            "Classic crisis pattern with pre-event complacency, dramatic VIX spike, and extended normalization period." if selected_event == "2008 Financial Crisis" else
            "Record-setting VIX spike from complacent levels, followed by relatively fast normalization as policy responses took effect." if selected_event == "2020 COVID Crash" else
            "Moderate VIX elevation reflecting policy uncertainty rather than systemic crisis, with a slower buildup and extended moderate-stress plateau."
        }
        """)
        
        # Trading implications
        st.info("""
        **Trading/Investment Implications:**
        
        When VIX spikes dramatically as in this example:
        
        1. **Short-term:** Extreme VIX readings often indicate capitulation and potential short-term bounces
        
        2. **Medium-term:** VIX normalization pattern can help gauge the speed of market recovery
        
        3. **Volatility Trading:** VIX term structure typically inverts during these events, creating potential opportunities in volatility products
        
        4. **Options Strategy:** High implied volatility makes option selling strategies potentially attractive, though with higher risk
        """)
    
    elif activity == "VIX Prediction Challenge":
        st.subheader("VIX Prediction Challenge")
        st.markdown("""
        Test your understanding of VIX dynamics by predicting future VIX levels based on different market scenarios.
        """)
        
        st.write("**Scenario 1: The Market Selloff**")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            The S&P 500 has fallen 12% over the last three weeks. Current metrics:
            - Realized volatility: 22%
            - Current VIX: 28%
            - Mean reversion level: 16%
            - Mean reversion speed: 0.25
            - Typical volatility premium: 4%
            
            If the market falls another 5% next week, what do you expect VIX to do?
            """)
            
            user_prediction = st.slider("Your VIX prediction:", 15, 50, 28, key="scenario1")
            
            if st.button("Check Answer", key="check1"):
                expected_vix = 22 + (16 - 22) * 0.25 + 4  # Realized vol + MR adjustment + premium
                expected_vix_up = expected_vix + 7  # Expected adjustment for 5% further drop
                
                st.markdown(f"""
                **Analysis:**
                - Expected VIX currently: {expected_vix:.1f}%
                - Current VIX vs Expected: {28 - expected_vix:.1f}% {("higher" if 28 > expected_vix else "lower")}
                - With 5% further drop, expected VIX: {expected_vix_up:.1f}%
                
                Your prediction of {user_prediction}% is {abs(user_prediction - expected_vix_up):.1f}% {("above" if user_prediction > expected_vix_up else "below")} 
                the analytical expectation.
                
                {'👍 Good prediction! A further market decline would likely push VIX higher, but since VIX is already elevated relative to realized volatility, the increase might be moderated.' if abs(user_prediction - expected_vix_up) < 5 else 'Consider that VIX tends to spike with market declines, but the relationship isn\'t linear, especially when VIX is already elevated.'}
                """)
        
        with col2:
            st.markdown("""
            **Scenario 2: The Relief Rally**
            
            After a period of uncertainty, the S&P 500 has rallied 8% over two weeks. Current metrics:
            - Realized volatility: 18%
            - Current VIX: 20%
            - Mean reversion level: 16%
            - Mean reversion speed: 0.25
            - Typical volatility premium: 4%
            
            If the market rises another 3% next week, what do you expect VIX to do?
            """)
            
            user_prediction2 = st.slider("Your VIX prediction:", 10, 30, 20, key="scenario2")
            
            if st.button("Check Answer", key="check2"):
                expected_vix = 18 + (16 - 18) * 0.25 + 4  # Realized vol + MR adjustment + premium
                expected_vix_down = expected_vix - 4  # Expected adjustment for 3% further rally
                
                st.markdown(f"""
                **Analysis:**
                - Expected VIX currently: {expected_vix:.1f}%
                - Current VIX vs Expected: {20 - expected_vix:.1f}% {("higher" if 20 > expected_vix else "lower")}
                - With 3% further rally, expected VIX: {expected_vix_down:.1f}%
                
                Your prediction of {user_prediction2}% is {abs(user_prediction2 - expected_vix_down):.1f}% {("above" if user_prediction2 > expected_vix_down else "below")} 
                the analytical expectation.
                
                {'👍 Good prediction! A relief rally typically leads to VIX declines as market fears subside. The speed of decline depends on how quickly realized volatility falls.' if abs(user_prediction2 - expected_vix_down) < 3 else 'Remember that VIX tends to fall as markets rally, and this decline can be substantial when previous uncertainty is resolved.'}
                """)
        
        # Challenge section
        st.subheader("Multi-Factor Challenge")
        
        st.markdown("""
        **Advanced Scenario: Mixed Signals**
        
        You're analyzing market conditions with conflicting signals:
        - The S&P 500 is up 2% this month, but down 5% from its all-time high
        - Realized volatility has been 14% over the past month
        - Current VIX is 17%
        - VIX term structure is flat (unusual)
        - Mean reversion level: 16%
        - Mean reversion speed: 0.25
        - Typical volatility premium: 4%
        - Fed meeting with potential policy changes next week
        
        Predict VIX one week from now, considering all factors.
        """)
        
        user_prediction3 = st.slider("Your VIX prediction:", 10, 30, 17, key="scenario3")
        fed_decision = st.radio("What do you think the Fed will do?", ["Hawkish surprise", "As expected", "Dovish surprise"])
        
        if st.button("Analyze Prediction", key="check3"):
            expected_vix = 14 + (16 - 14) * 0.25 + 4  # Realized vol + MR adjustment + premium
            
            fed_impact = 0
            if fed_decision == "Hawkish surprise":
                fed_impact = 4
            elif fed_decision == "Dovish surprise":
                fed_impact = -2
                
            expected_vix_final = expected_vix + fed_impact
            
            st.markdown(f"""
            **Multi-Factor Analysis:**
            - Base expected VIX: {expected_vix:.1f}%
            - Current VIX vs Expected: {17 - expected_vix:.1f}% {("higher" if 17 > expected_vix else "lower")}
            - Flat term structure suggests market uncertainty about direction
            - Fed impact (estimated): {fed_impact:.1f}%
            - Final expected VIX: {expected_vix_final:.1f}%
            
            Your prediction of {user_prediction3}% is {abs(user_prediction3 - expected_vix_final):.1f}% {("above" if user_prediction3 > expected_vix_final else "below")} 
            our model expectation.
            
            **Key insight:** In mixed signal environments, event-driven volatility (like Fed meetings) often dominates technical factors. The flat VIX term structure suggests the market is already pricing in some uncertainty about the Fed decision.
            """)
            
            if abs(user_prediction3 - expected_vix_final) < 3:
                st.success("Excellent analysis! You've effectively balanced multiple factors.")
            else:
                st.info("Consider how central bank policy surprises historically impact volatility. Markets price in expected outcomes, but react strongly to surprises.")
    
    else:  # Volatility Regime Quiz
        st.subheader("Volatility Regime Quiz")
        st.markdown("""
        Test your knowledge of volatility regimes and VIX interpretation with this quiz.
        """)
        
        # Initialize session state if needed
        if 'quiz_score' not in st.session_state:
            st.session_state.quiz_score = 0
            st.session_state.questions_answered = 0
            st.session_state.current_question = 0
        
        # Quiz questions
        questions = [
            {
                "question": "If VIX is at 12% and realized volatility is at 14%, this likely indicates:",
                "options": [
                    "Market fear", 
                    "Market complacency", 
                    "Normal market conditions", 
                    "Imminent market crash"
                ],
                "correct": 1,
                "explanation": "When VIX is below realized volatility, it often indicates market complacency, as investors aren't demanding the usual premium for protection."
            },
            {
                "question": "Which VIX level is typically considered the boundary between 'normal' and 'elevated' volatility?",
                "options": ["12%", "20%", "30%", "40%"],
                "correct": 1,
                "explanation": "The 20% level is commonly considered the threshold between normal volatility and elevated volatility or market stress."
            },
            {
                "question": "The 'volatility risk premium' in VIX refers to:",
                "options": [
                    "The commission paid to trade VIX futures", 
                    "The excess of VIX over subsequent realized volatility", 
                    "The cost of VIX options", 
                    "The spread between VIX and VXN (Nasdaq volatility index)"
                ],
                "correct": 1,
                "explanation": "The volatility risk premium is the typical excess of VIX (implied volatility) over the subsequently realized volatility, historically around 4-5 percentage points on average."
            },
            {
                "question": "If VIX is significantly higher than its 'expected' level based on recent realized volatility, this suggests:",
                "options": [
                    "Markets are likely to rally soon", 
                    "Implied volatility is too low", 
                    "Heightened market fear or anticipation of events", 
                    "VIX futures are in contango"
                ],
                "correct": 2,
                "explanation": "When VIX is significantly above its expected level (accounting for recent volatility and the typical premium), it indicates heightened market fear or anticipation of potential destabilizing events."
            },
            {
                "question": "Mean reversion in volatility suggests that:",
                "options": [
                    "VIX always returns to 20%", 
                    "Periods of extremely high or low volatility tend to be followed by moves toward average levels", 
                    "Option prices always converge to fair value", 
                    "Implied volatility equals historical volatility over time"
                ],
                "correct": 1,
                "explanation": "Volatility is mean-reverting, meaning that after periods of extremely high or low volatility, it tends to move back toward long-term average levels."
            },
            {
                "question": "VIX term structure is in 'backwardation' when:",
                "options": [
                    "VIX is below its historic average", 
                    "Short-term VIX futures are priced higher than longer-term futures", 
                    "VIX is trending downward", 
                    "Options are overpriced relative to fair value"
                ],
                "correct": 1,
                "explanation": "VIX term structure is in backwardation when near-term VIX futures trade at higher prices than longer-dated futures, typically during market stress when immediate volatility concerns outweigh longer-term ones."
            },
            {
                "question": "Which market regime typically has the highest volatility risk premium?",
                "options": [
                    "Bull markets with low volatility", 
                    "Bear markets with high volatility", 
                    "Sideways markets with moderate volatility", 
                    "The premium is constant across regimes"
                ],
                "correct": 1,
                "explanation": "The volatility risk premium often expands during high-volatility bear markets when demand for protection is highest, causing implied volatility (VIX) to exceed realized volatility by a greater margin."
            },
            {
                "question": "Expected VIX is calculated as:",
                "options": [
                    "Average VIX over the past year", 
                    "Realized volatility × 1.2", 
                    "Recent volatility + Mean reversion adjustment + Volatility premium", 
                    "Futures prices weighted by trading volume"
                ],
                "correct": 2,
                "explanation": "Expected VIX is calculated as the recent realized volatility, plus an adjustment for mean reversion, plus the typical volatility premium observed in the market."
            },
            {
                "question": "VIX rising while the market is also rising usually indicates:",
                "options": [
                    "A healthy bull market", 
                    "An imminent market reversal", 
                    "Increasing uncertainty despite positive returns", 
                    "A mathematical error in VIX calculation"
                ],
                "correct": 2,
                "explanation": "When VIX rises alongside equity prices (a rare divergence), it typically indicates increasing uncertainty or concern despite the positive returns, often preceding market corrections."
            },
            {
                "question": "Which of these factors does NOT directly impact VIX levels?",
                "options": [
                    "S&P 500 option prices", 
                    "Expected future volatility", 
                    "Current trading volume in the market", 
                    "Market fear and uncertainty"
                ],
                "correct": 2,
                "explanation": "While VIX is directly calculated from S&P 500 option prices (which reflect expected volatility, fear, and uncertainty), trading volume itself is not a direct input to the VIX calculation."
            }
        ]
        
        # Display current question
        if st.session_state.current_question < len(questions):
            current_q = questions[st.session_state.current_question]
            
            st.markdown(f"**Question {st.session_state.current_question + 1} of {len(questions)}:**")
            st.markdown(f"### {current_q['question']}")
            
            # Display options
            user_answer = st.radio("Select your answer:", current_q['options'], key=f"q{st.session_state.current_question}")
            selected_index = current_q['options'].index(user_answer)
            
            # Create columns for the answer submission and feedback
            col1, col2 = st.columns([1, 2])
            
            with col1:
                answer_submitted = st.button("Submit Answer", key=f"submit_{st.session_state.current_question}")
            
            # Create a variable to store if we've shown an answer
            if 'answer_shown' not in st.session_state:
                st.session_state.answer_shown = False
                
            if answer_submitted or st.session_state.answer_shown:
                st.session_state.answer_shown = True
                
                if selected_index == current_q['correct']:
                    st.success("✅ Correct!")
                    if not 'question_scored' in st.session_state or not st.session_state.question_scored:
                        st.session_state.quiz_score += 1
                        st.session_state.question_scored = True
                else:
                    st.error("❌ Incorrect")
                    
                st.info(f"**Explanation:** {current_q['explanation']}")
                st.session_state.questions_answered += 1
                
                if st.button("Next Question", key=f"next_{st.session_state.current_question}"):
                    st.session_state.current_question += 1
                    st.session_state.answer_shown = False
                    st.session_state.question_scored = False
                    st.rerun()
            
        else:
            # Quiz completed
            score_percentage = (st.session_state.quiz_score / len(questions)) * 100
            
            st.markdown(f"### Quiz Complete!")
            st.markdown(f"You scored: **{st.session_state.quiz_score}/{len(questions)}** ({score_percentage:.1f}%)")
            
            if score_percentage >= 80:
                st.success("🏆 Excellent! You have a strong understanding of VIX dynamics and interpretation.")
            elif score_percentage >= 60:
                st.info("👍 Good job! You have a solid foundation in VIX concepts, with room to refine your understanding.")
            else:
                st.warning("📚 Keep learning! Review the VIX theory and practice more with the interactive tools.")
            
            if st.button("Restart Quiz"):
                st.session_state.quiz_score = 0
                st.session_state.questions_answered = 0
                st.session_state.current_question = 0
                st.rerun()

# Modern UI-style disclaimer (Bootstrap-like "alert-danger")
st.markdown("""
<div style="
    background-color: #f8d7da; 
    color: #721c24; 
    padding: 20px; 
    border-radius: 8px; 
    margin-bottom: 20px;
">
  <h4 style="margin-top: 0;">
    <strong>IMPORTANT DISCLAIMER</strong>
  </h4>
  <ul style="list-style-type: disc; padding-left: 1.5em;">
    <li>VIX and volatility products are <em>complex investment tools</em> widely used by professional investors who typically have 
      <strong>many years of formal education and intensive training</strong>.</li>
    <li>Even these professionals often <strong>fail to outperform</strong> a simple buy-and-hold strategy 
      in a diversified index, as <strong>Warren Buffett</strong> and numerous 
      <strong>Nobel Prize-winning economists</strong> have demonstrated.</li>
    <li>The reality is that <strong>markets are smarter</strong> than any individual, making consistent 
      outperformance extremely difficult.</li>
    <li>If you'd like more insight into how challenging it is to "beat the market," explore reputable 
      <a href="#" style="color: #721c24; text-decoration: underline;">financial research</a> 
      on market efficiency.</li>
    <li>The author's main interest here is <strong>intellectual curiosity</strong> about the science and tools of finance, 
      not promoting active trading strategies.</li>
    <li>This material is <strong>purely educational</strong>. The author does <strong>not</strong> recommend any retail investor 
      engage in volatility trading.</li>
  </ul>
</div>
""", unsafe_allow_html=True)