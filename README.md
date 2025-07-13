# TradeML - Event-Driven Machine Learning Framework for Trading

TradeML is a Python framework for systematic trading strategy development based on the methodologies described in "Advances in Financial Machine Learning" by Marcos López de Prado. The framework implements an event-driven approach to generate labeled datasets for ML model training.

## Core Philosophy

The framework is built around **de Prado's event-based labeling methodology**:

1. **Events**: Identify points of interest in financial time series
2. **Barriers**: Apply generalized triple-barrier method to frame events  
3. **Trades**: Generate labeled training samples from barrier-bounded events
4. **ML Training**: Use Trades as a dataset for ML training

## Quick Start Example

```python
from data_process import DataLoader
from indicators import SimpleSupportResistance
from strategies import SimpleSRStrategy
from plotting import plot_market_data, Indicator
import pandas as pd

# Load data with simplified structure {symbol: DataFrame}
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT"],
    timeframe="1h",
    start_date="2017-12-01",
    end_date="2025-12-01",
    data_root="./data/crypto"
)

# Add indicators
indicators = [SimpleSupportResistance(lookback=20)]
for indicator in indicators:
    dfs = indicator.calculate(dfs, append=True)

# Generate events and trades
strategy = SimpleSRStrategy(params={
    'lookback': 20,
    'hold_periods': 100,
    'barrier_method': 'simple',
    'window': 40,
    'multiplier': [4, 4],
    'min_ret': 0.001
})

events = strategy.generate_events(dfs, set_barriers=True)
trades = strategy.generate_trades(dfs)

# Visualize with signals
symbol = "BTCUSDT"
df = dfs[symbol]

# Create signals DataFrame
signals_df = pd.DataFrame({
    'signal_values': trades[symbol].direction * trades[symbol].bin,
    'entry_price': events[symbol]['entry_price']
}, index=trades[symbol].index)

# Configure signals plotting
signals_config = {
    'prediction_quality': {
        'values_col': 'signal_values',    # Column with 0, 1, -1 values
        'price_col': 'entry_price'        # Column with price values
    }
}

# Create interactive plot
plot_market_data(
    df=df,
    indicators=[
        Indicator(df.SimpleSR_20_Resistance.values, name='SSR_20_Resistance', overlay=True, color='blue'),
        Indicator(df.SimpleSR_20_Support.values, name='SSR_20_Support', overlay=True, color='blue'),
        Indicator(df['Close'].pct_change().values, name='Returns', overlay=False)
    ],
    events=events[symbol],
    signals=signals_df,
    signals_config=signals_config,
    plot_width=1200,
    plot_volume=True,
    show_legend=True
)
```

## Central Architecture

### 🎯 Events as the Foundation

Events are the central concept here - they represent moments in time when something interesting happens in the market:

```python
# Events can be generated from various sources:
from events import StatisticalEvents, IndicatorEvents

# Statistical approach (e.g., CUSUM filters)
cusum_events = StatisticalEvents(threshold=0.05)
events = cusum_events.generate(price_data)

# Indicator-based approach  
sr_events = IndicatorEvents(indicator=support_resistance)
events = sr_events.generate(price_data)
```

### 🎪 Barrier Method (Generalized Triple-Barrier)

Each event is framed by barriers that define potential outcomes:

- **Profit-taking barrier** (upper): Target profit level
- **Stop-loss barrier** (lower): Maximum acceptable loss  
- **Vertical barrier** (time): Maximum holding period

```python
# Events are automatically wrapped with barriers
strategy = BaseStrategy(params={
    'pt_multiplier': 2.0,    # Profit-taking level
    'sl_multiplier': 1.0,    # Stop-loss level  
    'hold_periods': 50       # Time barrier
})

trades = strategy.generate_trades(price_data)
```

### 🏷️ Trade Generation for ML

The combination of events and barriers produces labeled training samples:

```python
# Each trade contains:
# - Entry conditions (indicators, market state)
# - Exit outcome (profit/loss/timeout)  
# - Features at entry time
# - Labels for supervised learning

trades = generate_trades(price_data, events)
# Result: DataFrame with features and labels ready for ML training
```

### 📊 Advanced Visualization with Signals

The framework supports plotting arbitrary signal columns alongside events and indicators:

```python
from plotting import plot_market_data

# Create signals DataFrame with values 0, 1, -1
signals_df = pd.DataFrame({
    'direction_bin': events['direction'] * trades['bin'],  # Direction * outcome
    'custom_signal': custom_signal_values,
    'price_signal': price_based_signals
}, index=events.index)

# Plot with signals
plot_market_data(
    df=price_data,
    events=events,
    signals=signals_df,  # New parameter for arbitrary signals
    indicators=indicators
)
# Result: Interactive plot with green diamonds (positive), red diamonds (negative)
```

## Workflow

### 1. Event Identification
Generate events using various methods:
- **Statistical**: CUSUM filters, volatility clustering, regime changes
- **Technical**: Indicator-based signals, pattern recognition
- **Microstructural**: Order flow imbalances, tick-based events

### 2. Barrier Application  
Apply the generalized triple-barrier method:
- Dynamic barrier sizing based on volatility
- Asymmetric barriers for directional strategies
- Custom barrier logic for specific use cases

### 3. Trade Labeling
Transform barrier-bounded events into training data:
- Binary labels (profit/loss)
- Continuous labels (returns) 
- Multi-class labels (profit/loss/timeout)

### 4. ML Model Training
Use generated trades for machine learning:
- Feature engineering from market indicators
- Label quality assessment and filtering
- Model training with properly labeled data

## Key Components

### Event Generators
```python
from events.base import EventGenerator, EventType

class CustomEventGenerator(EventGenerator):
    def generate(self, data, **kwargs):
        # Implement your event logic
        # Return timestamps where events occur
        return event_timestamps
```

### Strategy Framework  
```python
from strategies.base_strategy import BaseStrategy

class MLStrategy(BaseStrategy):
    def _generate_raw_events(self, data):
        # Define when to enter trades
        return events
        
    def init(self):
        # Add indicators and event generators
        pass
```

### Barrier Management
The framework implements de Prado's enhanced barrier method:
- **Symmetric barriers**: Equal profit/loss thresholds
- **Asymmetric barriers**: Different upside/downside targets
- **Dynamic barriers**: Volatility-adjusted levels
- **Meta-barriers**: Barriers based on other events

## Why This Approach?

Traditional approaches to creating training data for financial ML suffer from:
- **Low signal-to-noise ratio**: Random sampling doesn't focus on interesting periods
- **Look-ahead bias**: Using future information in feature construction
- **Label ambiguity**: Unclear definition of success/failure

The event-driven approach addresses these issues by:
- **Focusing on meaningful moments**: Events highlight periods of interest
- **Clear labeling**: Barriers provide unambiguous success/failure criteria  
- **Reduced overfitting**: Less correlated samples improve generalization

## Current Status

⚠️ **This package is under active development**

The framework currently implements:
- ✅ Event generation system
- ✅ Trade generation pipeline
- ✅ Basic indicator integration
- ✅ Crypto data loader (Binance API)
- ✅ Parquet data storage and loading
- ✅ Advanced visualization with signals plotting
- ✅ Interactive Bokeh-based plotting
- 🚧 ML model integration (planned)
- 🚧 Generalized barrier method
- 🚧 Advanced event generators (expanding)
- 🚧 Performance analytics (planned)

## Quick Setup

### Prerequisites
- Anaconda or Miniconda installed
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trademl
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate environment:**
   ```bash
   conda activate trademl
   ```

4. **Install additional dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

1. **Download cryptocurrency data:**
   ```bash
   python data_process/download_full_history.py --count 10 --intervals 1h
   ```

2. **Or use the data loader in Python:**
   ```python
   from data_process import DataLoader
   
   # Load data for multiple symbols
   data = DataLoader.load_crypto_data_single_timeframe(
       symbols=["BTCUSDT", "ETHUSDT"],
       timeframe="1h",
       data_root="data/crypto"
   )
   ```

## Advanced Example

```python
from strategies.base_strategy import BaseStrategy
from events.statistical import CUSUMEvents

# Create CUSUM-based event generator
cusum_events = CUSUMEvents(threshold=0.02)

# Define strategy with events
class CUSUMStrategy(BaseStrategy):
    def init(self):
        self.event_generator = cusum_events
        
    def _generate_raw_events(self, data):
        return self.event_generator.generate(data)

# Generate labeled training data
strategy = CUSUMStrategy()
trades = strategy.generate_trades(price_data)

# Now 'trades' contains features and labels for ML training
```

## Future Roadmap

- **Enhanced Event Generators**: More sophisticated statistical and technical event detection
- **ML Integration**: Direct integration with scikit-learn, XGBoost, and deep learning frameworks  
- **Portfolio Construction**: Multi-asset event-driven strategies
- **Backtesting Engine**: Comprehensive strategy evaluation tools
- **Risk Management**: Position sizing and portfolio-level risk controls


## References

This framework implements concepts from:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
