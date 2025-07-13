#!/usr/bin/env python3
"""
Download full historical data for top USDT cryptocurrency pairs (parallel version).
"""

from data_process.crypto_data_loader import download_crypto_data_parallel
from data_process.binance_symbols import BinanceSymbolsFetcher
import argparse
from datetime import datetime, timedelta
import time
from tqdm import tqdm

def get_earliest_available_date():
    return "2017-08-01"

def estimate_download_time(symbols, intervals, start_date, end_date, workers):
    total_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    requests_per_interval = {}
    for interval in intervals:
        if interval == '1m':
            candles_per_day = 1440
        elif interval == '5m':
            candles_per_day = 288
        elif interval == '1h':
            candles_per_day = 24
        elif interval == '1d':
            candles_per_day = 1
        else:
            candles_per_day = 24
        total_candles = total_days * candles_per_day
        requests_per_interval[interval] = max(1, total_candles // 1000)
    total_requests = sum(requests_per_interval.values()) * len(symbols)
    estimated_seconds = total_requests * 0.5 / max(1, workers)
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    return hours, minutes

def main():
    parser = argparse.ArgumentParser(description='Download full historical data for top USDT pairs (parallel)')
    parser.add_argument('--count', type=int, default=20, help='Number of top USDT pairs to download (default: 20)')
    parser.add_argument('--intervals', nargs='+', default=['1m', '5m', '1h', '1d'], help='Time intervals to download (default: 1m, 5m, 1h, 1d)')
    parser.add_argument('--start-date', default=None, help='Start date in YYYY-MM-DD format (default: earliest available)')
    parser.add_argument('--end-date', default=None, help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--save-root', default='data/crypto', help='Root directory for saving data (default: data/crypto)')
    parser.add_argument('--estimate-only', action='store_true', help='Only estimate download time without downloading')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    fetcher = BinanceSymbolsFetcher()
    symbols = fetcher.get_top_usdt_pairs(args.count)
    if not args.start_date:
        args.start_date = get_earliest_available_date()
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    print("🚀 Full Historical Data Download (Parallel)")
    print("=" * 50)
    print(f"📊 Symbols: {symbols}")
    print(f"📅 Date range: {args.start_date} to {args.end_date}")
    print(f"⏰ Intervals: {args.intervals}")
    print(f"💾 Save root: {args.save_root}")
    print(f"🔄 Overwrite: {args.overwrite}")
    print(f"🧵 Workers: {args.workers}")

    hours, minutes = estimate_download_time(symbols, args.intervals, args.start_date, args.end_date, args.workers)
    print(f"⏱️  Estimated download time: {int(hours)}h {int(minutes)}m")

    if args.estimate_only:
        print("\n✅ Estimation complete. Use --estimate-only to see download time without downloading.")
        return

    print(f"\n⚠️  This will download {len(symbols)} symbols × {len(args.intervals)} intervals in parallel")
    print(f"   Estimated time: {int(hours)}h {int(minutes)}m")
    response = input("\nContinue? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    print("\n🚀 Starting parallel download...")
    start_time = time.time()
    
    # Calculate total tasks for progress tracking
    total_tasks = len(symbols) * len(args.intervals)
    print(f"📊 Total tasks to process: {total_tasks}")
    print(f"⏱️  Progress will be shown below:")
    print("-" * 50)
    
    results = download_crypto_data_parallel(
        symbols=symbols,
        intervals=args.intervals,
        start_date=args.start_date,
        end_date=args.end_date,
        save_root=args.save_root,
        overwrite=args.overwrite,
        max_workers=args.workers
    )
    
    actual_time = time.time() - start_time
    actual_hours = actual_time // 3600
    actual_minutes = (actual_time % 3600) // 60

    print("\n📊 Download Results:")
    print("=" * 40)
    total_success = 0
    total_failed = 0
    
    # Show results with progress bar
    total_results = sum(len(symbol_results) for symbol_results in results.values())
    with tqdm(total=total_results, desc="Processing results", unit="result") as pbar:
        for interval, symbol_results in results.items():
            print(f"\n{interval.upper()}:")
            interval_success = 0
            interval_failed = 0
            for symbol, success in symbol_results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"  {symbol:<12} {status}")
                if success:
                    interval_success += 1
                    total_success += 1
                else:
                    interval_failed += 1
                    total_failed += 1
                pbar.update(1)
            print(f"  Summary: {interval_success} success, {interval_failed} failed")
    
    print(f"\n🎯 Overall Summary:")
    print(f"  Total successful downloads: {total_success}")
    print(f"  Total failed downloads: {total_failed}")
    print(f"  Success rate: {(total_success/(total_success+total_failed)*100):.1f}%")
    print(f"  Actual time taken: {int(actual_hours)}h {int(actual_minutes)}m")
    if total_failed > 0:
        print(f"\n⚠️  Some downloads failed. Check the logs above for details.")
    print(f"\n✅ Download complete! Data saved to {args.save_root}/")
    print(f"📁 Files created:")
    for interval in args.intervals:
        print(f"   • {args.save_root}/{interval}/ - {interval} data")

if __name__ == "__main__":
    main() 