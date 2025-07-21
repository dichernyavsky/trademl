import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def quick_trades_analysis(trades: Union[Dict[str, pd.DataFrame], pd.DataFrame], 
                         round_digits: int = 2, show_all: bool = True):
    """
    Quick analysis of trading performance with all key metrics.
    
    Args:
        trades: Dictionary of trades {symbol: DataFrame} or single DataFrame
        round_digits: Number of decimal places to round to
        show_all: Whether to show all metrics (True) or just win/loss ratio (False)
    """
    if isinstance(trades, dict):
        for key in trades.keys():
            if len(trades[key]) > 0:
                profit_trades = (trades[key]['bin'] == 1).sum()
                loss_trades = (trades[key]['bin'] == -1).sum()
                time_barrier_trades = (trades[key]['bin'] == 0).sum()
                total_trades = len(trades[key])
                
                # Calculate ratios
                win_loss_ratio = loss_trades / profit_trades if profit_trades > 0 else float('inf')
                time_barrier_ratio = time_barrier_trades / total_trades
                loss_plus_time_ratio = (loss_trades + time_barrier_trades) / profit_trades if profit_trades > 0 else float('inf')
                
                if show_all:
                    print(f"{key}: Win/Loss={round(win_loss_ratio, round_digits)}, "
                          f"Time={round(time_barrier_ratio, round_digits)}, "
                          f"(Loss+Time)/Profit={round(loss_plus_time_ratio, round_digits)}")
                else:
                    print(f"{key}: {round(win_loss_ratio, round_digits)}")
    else:
        # Single DataFrame
        if len(trades) > 0:
            profit_trades = (trades['bin'] == 1).sum()
            loss_trades = (trades['bin'] == -1).sum()
            time_barrier_trades = (trades['bin'] == 0).sum()
            total_trades = len(trades)
            
            # Calculate ratios
            win_loss_ratio = loss_trades / profit_trades if profit_trades > 0 else float('inf')
            time_barrier_ratio = time_barrier_trades / total_trades
            loss_plus_time_ratio = (loss_trades + time_barrier_trades) / profit_trades if profit_trades > 0 else float('inf')
            
            if show_all:
                print(f"Single: Win/Loss={round(win_loss_ratio, round_digits)}, "
                      f"Time={round(time_barrier_ratio, round_digits)}, "
                      f"(Loss+Time)/Profit={round(loss_plus_time_ratio, round_digits)}")
            else:
                print(f"Single: {round(win_loss_ratio, round_digits)}")

class TradesAnalyzer:
    """
    Analyzer class for trading performance statistics.
    
    This class provides comprehensive analysis of trades including:
    - Win/Loss ratios
    - Time barrier hits
    - Overall performance metrics
    - Visualizations
    """
    
    def __init__(self, trades: Union[Dict[str, pd.DataFrame], pd.DataFrame]):
        """
        Initialize the trades analyzer.
        
        Args:
            trades: Dictionary of trades {symbol: DataFrame} or single DataFrame
        """
        self.trades = trades
        self.stats = {}
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate all trading statistics."""
        if isinstance(self.trades, dict):
            for symbol, trades_df in self.trades.items():
                if len(trades_df) > 0:
                    self.stats[symbol] = self._calculate_symbol_stats(trades_df)
        else:
            # Single DataFrame
            if len(self.trades) > 0:
                self.stats['single'] = self._calculate_symbol_stats(self.trades)
    
    def _calculate_symbol_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate statistics for a single symbol."""
        if 'bin' not in trades_df.columns:
            raise ValueError("Trades DataFrame must contain 'bin' column")
        
        total_trades = len(trades_df)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'time_barrier_rate': 0.0,
                'win_loss_ratio': 0.0,
                'loss_plus_time_ratio': 0.0,
                'profit_trades': 0,
                'loss_trades': 0,
                'time_barrier_trades': 0
            }
        
        # Count different trade outcomes
        profit_trades = (trades_df['bin'] == 1).sum()
        loss_trades = (trades_df['bin'] == -1).sum()
        time_barrier_trades = (trades_df['bin'] == 0).sum()
        
        # Calculate rates
        win_rate = profit_trades / total_trades
        loss_rate = loss_trades / total_trades
        time_barrier_rate = time_barrier_trades / total_trades
        
        # Calculate ratios
        win_loss_ratio = loss_trades / profit_trades if profit_trades > 0 else float('inf')
        loss_plus_time_ratio = (loss_trades + time_barrier_trades) / profit_trades if profit_trades > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'time_barrier_rate': time_barrier_rate,
            'win_loss_ratio': win_loss_ratio,
            'loss_plus_time_ratio': loss_plus_time_ratio,
            'profit_trades': profit_trades,
            'loss_trades': loss_trades,
            'time_barrier_trades': time_barrier_trades
        }
    
    def print_summary(self, round_digits: int = 2):
        """
        Print a comprehensive summary of trading statistics.
        
        Args:
            round_digits: Number of decimal places to round to
        """
        print("=" * 60)
        print("TRADING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for symbol, stats in self.stats.items():
            print(f"\n📊 {symbol.upper()}")
            print("-" * 40)
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Profit Trades: {stats['profit_trades']} ({stats['win_rate']:.1%})")
            print(f"Loss Trades: {stats['loss_trades']} ({stats['loss_rate']:.1%})")
            print(f"Time Barrier Trades: {stats['time_barrier_trades']} ({stats['time_barrier_rate']:.1%})")
            print()
            print(f"Win/Loss Ratio: {stats['win_loss_ratio']:.{round_digits}f}")
            print(f"(Loss + Time Barrier) / Profit Ratio: {stats['loss_plus_time_ratio']:.{round_digits}f}")
    
    def get_win_loss_ratios(self, round_digits: int = 2) -> Dict[str, float]:
        """
        Get win/loss ratios for all symbols.
        
        Args:
            round_digits: Number of decimal places to round to
            
        Returns:
            Dictionary of {symbol: win_loss_ratio}
        """
        ratios = {}
        for symbol, stats in self.stats.items():
            ratios[symbol] = round(stats['win_loss_ratio'], round_digits)
        return ratios
    
    def get_time_barrier_ratios(self, round_digits: int = 2) -> Dict[str, float]:
        """
        Get time barrier ratios for all symbols.
        
        Args:
            round_digits: Number of decimal places to round to
            
        Returns:
            Dictionary of {symbol: time_barrier_ratio}
        """
        ratios = {}
        for symbol, stats in self.stats.items():
            ratios[symbol] = round(stats['time_barrier_rate'], round_digits)
        return ratios
    
    def get_loss_plus_time_ratios(self, round_digits: int = 2) -> Dict[str, float]:
        """
        Get (loss + time barrier) / profit ratios for all symbols.
        
        Args:
            round_digits: Number of decimal places to round to
            
        Returns:
            Dictionary of {symbol: ratio}
        """
        ratios = {}
        for symbol, stats in self.stats.items():
            ratios[symbol] = round(stats['loss_plus_time_ratio'], round_digits)
        return ratios
    
    def plot_performance_summary(self, figsize: tuple = (15, 10)):
        """
        Create a comprehensive visualization of trading performance.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.stats:
            print("No statistics available to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Trading Performance Analysis', fontsize=16, fontweight='bold')
        
        symbols = list(self.stats.keys())
        
        # 1. Win/Loss/Time Barrier Distribution
        ax1 = axes[0, 0]
        win_rates = [self.stats[s]['win_rate'] for s in symbols]
        loss_rates = [self.stats[s]['loss_rate'] for s in symbols]
        time_rates = [self.stats[s]['time_barrier_rate'] for s in symbols]
        
        x = np.arange(len(symbols))
        width = 0.25
        
        ax1.bar(x - width, win_rates, width, label='Profit', color='green', alpha=0.7)
        ax1.bar(x, loss_rates, width, label='Loss', color='red', alpha=0.7)
        ax1.bar(x + width, time_rates, width, label='Time Barrier', color='orange', alpha=0.7)
        
        ax1.set_xlabel('Symbols')
        ax1.set_ylabel('Rate')
        ax1.set_title('Trade Outcome Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(symbols, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss Ratios
        ax2 = axes[0, 1]
        win_loss_ratios = [self.stats[s]['win_loss_ratio'] for s in symbols]
        ax2.bar(symbols, win_loss_ratios, color='blue', alpha=0.7)
        ax2.set_xlabel('Symbols')
        ax2.set_ylabel('Win/Loss Ratio')
        ax2.set_title('Win/Loss Ratios')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Total Trades
        ax3 = axes[1, 0]
        total_trades = [self.stats[s]['total_trades'] for s in symbols]
        ax3.bar(symbols, total_trades, color='purple', alpha=0.7)
        ax3.set_xlabel('Symbols')
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Total Number of Trades')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. (Loss + Time) / Profit Ratios
        ax4 = axes[1, 1]
        loss_plus_time_ratios = [self.stats[s]['loss_plus_time_ratio'] for s in symbols]
        ax4.bar(symbols, loss_plus_time_ratios, color='brown', alpha=0.7)
        ax4.set_xlabel('Symbols')
        ax4.set_ylabel('(Loss + Time) / Profit Ratio')
        ax4.set_title('(Loss + Time Barrier) / Profit Ratios')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_detailed_stats(self) -> Dict:
        """
        Get all detailed statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return self.stats.copy() 