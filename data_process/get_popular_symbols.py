#!/usr/bin/env python3
"""
Quick script to get popular cryptocurrency symbols from Binance.
"""

from data_process.binance_symbols import get_popular_symbols, get_usdt_pairs, save_symbols_list

def main():
    print("🚀 Getting Popular Cryptocurrency Symbols")
    print("=" * 50)
    
    # Get popular symbols
    popular = get_popular_symbols()
    
    print(f"\n🌟 Popular Symbols ({len(popular)}):")
    for i, symbol in enumerate(popular, 1):
        print(f"  {i:2d}. {symbol}")
    
    # Get all USDT pairs
    usdt_pairs = get_usdt_pairs()
    print(f"\n📊 USDT Pairs Summary:")
    print(f"  Total USDT pairs: {len(usdt_pairs)}")
    print(f"  First 10: {usdt_pairs[:10]}")
    print(f"  Last 10: {usdt_pairs[-10:]}")
    
    # Save to files
    print(f"\n💾 Saving symbol lists...")
    save_symbols_list("../data/binance_symbols.csv")
    
    # Save popular symbols to text file
    with open("../data/popular_symbols.txt", "w") as f:
        f.write("\n".join(popular))
    
    # Save USDT pairs to text file
    with open("../data/usdt_pairs.txt", "w") as f:
        f.write("\n".join(usdt_pairs))
    
    print(f"✓ Files saved:")
    print(f"  • ../data/binance_symbols.csv - Complete symbols list")
    print(f"  • ../data/popular_symbols.txt - Popular symbols only")
    print(f"  • ../data/usdt_pairs.txt - All USDT pairs")
    
    print(f"\n🎯 Ready for data download!")
    print(f"Use these symbols in your download scripts:")

if __name__ == "__main__":
    main() 