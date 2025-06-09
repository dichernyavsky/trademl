# TradeML - Event-Driven Machine Learning Framework for Trading

TradeML is a Python framework for systematic trading strategy development based on the methodologies described in "Advances in Financial Machine Learning" by Marcos López de Prado. The framework implements an event-driven approach to generate labeled datasets for ML model training.

## Core Philosophy

The framework is built around **de Prado's event-based labeling methodology**:

1. **Events**: Identify points of interest in financial time series
2. **Barriers**: Apply generalized triple-barrier method to frame events  
3. **Trades**: Generate labeled training samples from barrier-bounded events
4. **ML Training**: Use Trades as a dataset for ML training


## Central Architecture

### 🎯 Events as the Foundation

Events are the central concept here - they represent moments in time when something interesting happens in the market:

```python
# Events can be generated from various sources:
from trademl.events import StatisticalEvents, IndicatorEvents

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
from trademl.events.base import EventGenerator, EventType

class CustomEventGenerator(EventGenerator):
    def generate(self, data, **kwargs):
        # Implement your event logic
        # Return timestamps where events occur
        return event_timestamps
```

### Strategy Framework  
```python
from trademl.strategies.base_strategy import BaseStrategy

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
- 🚧 ML model integration (planned)
- 🚧 Generalized barrier method
- 🚧 Advanced event generators (expanding)
- 🚧 Performance analytics (planned)

## Quick Example

```python
from trademl.strategies.base_strategy import BaseStrategy
from trademl.events.statistical import CUSUMEvents

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
