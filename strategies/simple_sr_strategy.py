from .base_strategy import BaseStrategy
# NOTE: changed for local testing in Jupyter
from ..events.indicator_events import SimpleSREventGenerator
#from events.indicator_events import SimpleSREventGenerator



class SimpleSREventStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(params)
        # Extract parameters for the event generator
        generator_params = {
            'lookback': self.params.get('lookback', 20),
            'mode': self.params.get('mode', 'breakout')
        }
        self.event_generator = SimpleSREventGenerator(**generator_params)
    
    def _generate_raw_events(self, data):
        """
        Generate raw SR breakout events for a single symbol.
        """
        return self.event_generator.generate(data, include_entry_price=True)

