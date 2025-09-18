import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

from dotenv import load_dotenv
from huggingface_hub import login

from predibench.agent.models import ModelInfo, ModelInvestmentDecisions
from predibench.invest import run_investments_for_specific_date
from predibench.logger_config import get_logger
from predibench.common import get_date_output_path

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

logger = get_logger(__name__)


def get_run_result_path(model_id: str, target_date: date, run_idx: int) -> Path:
    """Get the path where a specific run result should be saved."""
    date_output_path = get_date_output_path(target_date)
    run_result_path = date_output_path / f"{model_id.replace('/', '--')}_run_{run_idx}"
    run_result_path.mkdir(parents=True, exist_ok=True)
    return run_result_path


def check_run_exists(model_id: str, target_date: date, run_idx: int) -> bool:
    """Check if a run already exists by looking for the saved model result."""
    run_result_path = get_run_result_path(model_id, target_date, run_idx)
    result_file = run_result_path / "model_investment_decisions.json"
    return result_file.exists()


def load_existing_run(model_id: str, target_date: date, run_idx: int) -> ModelInvestmentDecisions:
    """Load an existing run from saved results."""
    run_result_path = get_run_result_path(model_id, target_date, run_idx)
    result_file = run_result_path / "model_investment_decisions.json"

    with open(result_file, 'r') as f:
        data = json.load(f)

    return ModelInvestmentDecisions(**data)


def move_result_to_run_location(model_info: ModelInfo, target_date: date, run_idx: int):
    """Move the model result from default location to run-specific location."""
    # Original location where the result was saved
    original_path = model_info.get_model_result_path(target_date)
    original_file = original_path / "model_investment_decisions.json"

    # New location for this specific run
    run_path = get_run_result_path(model_info.model_id, target_date, run_idx)
    run_file = run_path / "model_investment_decisions.json"

    if original_file.exists():
        import shutil
        shutil.move(str(original_file), str(run_file))


def run_multiple_experiments(
    model_info: ModelInfo,
    num_runs: int = 10,
    num_events: int = 10,
    target_date: date = None,
    force_rewrite: bool = False
) -> List[ModelInvestmentDecisions]:
    """Run the same LLM on the same events multiple times."""
    if target_date is None:
        target_date = date.today()

    logger.info(f"Running {num_runs} experiments with {model_info.model_pretty_name}")

    all_results = []

    for run_idx in range(num_runs):
        # Check if this run already exists
        if not force_rewrite and check_run_exists(model_info.model_id, target_date, run_idx):
            logger.info(f"Run {run_idx + 1}/{num_runs} already exists, loading from cache")
            existing_result = load_existing_run(model_info.model_id, target_date, run_idx)
            all_results.append(existing_result)
            continue

        logger.info(f"Starting run {run_idx + 1}/{num_runs}")

        # Use the same model_info but run with force_rewrite=True to overwrite previous result
        results = run_investments_for_specific_date(
            models=[model_info],
            max_n_events=num_events,
            target_date=target_date,
            time_until_ending=timedelta(days=7 * 6),
            force_rewrite=True,  # Always overwrite to get fresh results
            filter_crypto_events=True,
        )

        if results:
            # Move the result to run-specific location
            move_result_to_run_location(model_info, target_date, run_idx)

            # Load the result from the new location
            run_result = load_existing_run(model_info.model_id, target_date, run_idx)
            all_results.append(run_result)

    return all_results


def extract_decision_metrics(results_data: List[ModelInvestmentDecisions]) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]], Dict[str, List[Dict[str, Any]]]]:
    """Extract key metrics from all runs for analysis.

    Returns:
        - market_metrics: All market-level decisions aggregated
        - event_metrics: Event-level aggregated metrics
        - market_by_event_metrics: Market decisions grouped by event
        - market_details: Detailed market information for analysis
    """
    # Market-level metrics (all individual market decisions)
    market_metrics = {
        'odds': [],
        'bets': [],
        'confidence': []
    }

    # Event-level metrics (aggregated per event)
    event_metrics = {
        'total_allocated': [],
        'unallocated_capital': [],
        'num_positive_bets': [],
        'num_negative_bets': [],
        'avg_bet_magnitude': [],
        'num_markets_per_event': []
    }

    # Market decisions grouped by event for consistency analysis
    market_by_event_metrics = defaultdict(lambda: defaultdict(list))

    # Detailed market information for market-specific analysis
    market_details = defaultdict(list)

    for result in results_data:
        for event_decision in result.event_investment_decisions:
            event_id = event_decision.event_id

            total_allocated = 0
            positive_bets = 0
            negative_bets = 0
            bet_magnitudes = []

            # Process each market in this event
            for market_decision in event_decision.market_investment_decisions:
                decision = market_decision.decision
                market_id = market_decision.market_id

                # Collect individual market-level metrics
                market_metrics['odds'].append(decision.estimated_probability)
                market_metrics['bets'].append(decision.bet)
                market_metrics['confidence'].append(decision.confidence)

                # Group by event for consistency analysis
                market_by_event_metrics[event_id]['odds'].append(decision.estimated_probability)
                market_by_event_metrics[event_id]['bets'].append(decision.bet)
                market_by_event_metrics[event_id]['confidence'].append(decision.confidence)

                # Store detailed market information
                market_details[market_id].append({
                    'event_id': event_id,
                    'event_title': event_decision.event_title,
                    'market_question': market_decision.market_question,
                    'odds': decision.estimated_probability,
                    'bet': decision.bet,
                    'confidence': decision.confidence,
                    'run_index': len(market_details[market_id])
                })

                # Aggregate for event-level metrics
                total_allocated += abs(decision.bet)
                bet_magnitudes.append(abs(decision.bet))

                if decision.bet > 0:
                    positive_bets += 1
                elif decision.bet < 0:
                    negative_bets += 1

            # Store event-level aggregated metrics
            event_metrics['total_allocated'].append(total_allocated)
            event_metrics['unallocated_capital'].append(event_decision.unallocated_capital)
            event_metrics['num_positive_bets'].append(positive_bets)
            event_metrics['num_negative_bets'].append(negative_bets)
            event_metrics['avg_bet_magnitude'].append(np.mean(bet_magnitudes) if bet_magnitudes else 0)
            event_metrics['num_markets_per_event'].append(len(event_decision.market_investment_decisions))

    return market_metrics, event_metrics, dict(market_by_event_metrics), dict(market_details)


def calculate_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate descriptive statistics for each metric."""
    stats = {}

    for metric_name, values in metrics.items():
        if not values:
            continue

        stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'variance': np.var(values),
            'skewness': calculate_skewness(values),
            'kurtosis': calculate_kurtosis(values)
        }

    return stats


def calculate_skewness(values: List[float]) -> float:
    """Calculate skewness of the distribution."""
    if len(values) < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return np.mean([(x - mean) / std for x in values]) ** 3


def calculate_kurtosis(values: List[float]) -> float:
    """Calculate kurtosis of the distribution."""
    if len(values) < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return np.mean([(x - mean) / std for x in values]) ** 4 - 3


def create_market_behavior_overview(market_metrics: Dict[str, List[float]], model_name: str, output_dir: Path):
    """Create overview of market-level decision patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Market Decision Patterns: {model_name}', fontsize=16, fontweight='bold')

    # Estimated probability distribution
    ax = axes[0, 0]
    if 'odds' in market_metrics and market_metrics['odds']:
        values = market_metrics['odds']
        ax.hist(values, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
        ax.set_title('Market Estimated Probabilities')
        ax.set_xlabel('Estimated Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Bet amount distribution
    ax = axes[0, 1]
    if 'bets' in market_metrics and market_metrics['bets']:
        values = market_metrics['bets']
        ax.hist(values, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.set_title('Market Bet Amounts')
        ax.set_xlabel('Bet Amount')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Confidence distribution
    ax = axes[1, 0]
    if 'confidence' in market_metrics and market_metrics['confidence']:
        values = market_metrics['confidence']
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.set_title('Market Confidence Levels')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Bet magnitude vs confidence scatter
    ax = axes[1, 1]
    if ('bets' in market_metrics and 'confidence' in market_metrics and
        len(market_metrics['bets']) == len(market_metrics['confidence'])):
        bet_magnitudes = [abs(bet) for bet in market_metrics['bets']]
        ax.scatter(market_metrics['confidence'], bet_magnitudes, alpha=0.6, s=30)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Bet Magnitude (Absolute)')
        ax.set_title('Confidence vs Bet Magnitude')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'market_behavior_overview.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_event_capital_allocation(event_metrics: Dict[str, List[float]], model_name: str, output_dir: Path):
    """Create visualizations focusing on how capital is allocated across events."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Event Capital Allocation Patterns: {model_name}', fontsize=16, fontweight='bold')

    # Total allocated per event
    ax = axes[0, 0]
    if 'total_allocated' in event_metrics and event_metrics['total_allocated']:
        values = event_metrics['total_allocated']
        ax.hist(values, bins=15, alpha=0.7, edgecolor='black', color='gold')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.1f}')
        ax.set_title('Total Capital Allocated per Event')
        ax.set_xlabel('Total Allocated Amount')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Unallocated capital per event
    ax = axes[0, 1]
    if 'unallocated_capital' in event_metrics and event_metrics['unallocated_capital']:
        values = event_metrics['unallocated_capital']
        ax.hist(values, bins=15, alpha=0.7, edgecolor='black', color='lightblue')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.1f}')
        ax.set_title('Unallocated Capital per Event')
        ax.set_xlabel('Unallocated Amount')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Number of markets per event
    ax = axes[1, 0]
    if 'num_markets_per_event' in event_metrics and event_metrics['num_markets_per_event']:
        values = event_metrics['num_markets_per_event']
        unique_values = sorted(set(values))
        counts = [values.count(v) for v in unique_values]
        ax.bar(unique_values, counts, alpha=0.7, edgecolor='black', color='lightgreen')
        ax.set_title('Markets per Event Distribution')
        ax.set_xlabel('Number of Markets')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Allocation vs unallocated scatter
    ax = axes[1, 1]
    if ('total_allocated' in event_metrics and 'unallocated_capital' in event_metrics and
        len(event_metrics['total_allocated']) == len(event_metrics['unallocated_capital'])):
        ax.scatter(event_metrics['total_allocated'], event_metrics['unallocated_capital'],
                  alpha=0.6, s=40, color='purple')
        ax.set_xlabel('Total Allocated')
        ax.set_ylabel('Unallocated Capital')
        ax.set_title('Allocated vs Unallocated Capital')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'event_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_probability_bet_analysis(market_metrics: Dict[str, List[float]], model_name: str, output_dir: Path):
    """Create detailed analysis of probability estimates vs betting behavior."""
    if 'bets' not in market_metrics or not market_metrics['bets']:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Probability vs Betting Analysis: {model_name}', fontsize=16, fontweight='bold')

    bets = np.array(market_metrics['bets'])

    # Bet direction pie chart
    ax = axes[0, 0]
    positive_bets = bets[bets > 0]
    negative_bets = bets[bets < 0]
    zero_bets = bets[bets == 0]

    sizes = [len(positive_bets), len(negative_bets), len(zero_bets)]
    labels = ['Long Positions', 'Short Positions', 'No Position']
    colors = ['#2E8B57', '#DC143C', '#696969']

    non_zero_sizes = [s for s in sizes if s > 0]
    non_zero_labels = [labels[i] for i, s in enumerate(sizes) if s > 0]
    non_zero_colors = [colors[i] for i, s in enumerate(sizes) if s > 0]

    if non_zero_sizes:
        ax.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
              autopct='%1.1f%%', startangle=90)
        ax.set_title('Position Distribution')

    # Probability vs bet scatter with color coding
    ax = axes[0, 1]
    if 'odds' in market_metrics and len(market_metrics['odds']) == len(market_metrics['bets']):
        colors = ['red' if bet < 0 else 'green' if bet > 0 else 'gray' for bet in bets]
        ax.scatter(market_metrics['odds'], market_metrics['bets'], alpha=0.6, c=colors, s=30)
        ax.set_xlabel('Estimated Probability')
        ax.set_ylabel('Bet Amount')
        ax.set_title('Probability vs Bet Amount')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.grid(True, alpha=0.3)

        # Add quadrant labels
        ax.text(0.8, max(bets)*0.8, 'High Prob\nLong', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.2, max(bets)*0.8, 'Low Prob\nLong', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(0.8, min(bets)*0.8, 'High Prob\nShort', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.text(0.2, min(bets)*0.8, 'Low Prob\nShort', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Bet magnitude by probability bins
    ax = axes[1, 0]
    if 'odds' in market_metrics:
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        bet_magnitudes_by_bin = [[] for _ in range(len(bin_labels))]

        for prob, bet in zip(market_metrics['odds'], bets):
            bin_idx = min(int(prob * 5), 4)  # Ensure we don't exceed bin count
            bet_magnitudes_by_bin[bin_idx].append(abs(bet))

        # Filter out empty bins
        non_empty_bins = [(i, magnitudes) for i, magnitudes in enumerate(bet_magnitudes_by_bin) if magnitudes]

        if non_empty_bins:
            indices, magnitudes_list = zip(*non_empty_bins)
            labels = [bin_labels[i] for i in indices]
            ax.boxplot(magnitudes_list, labels=labels)
            ax.set_title('Bet Magnitude by Probability Range')
            ax.set_xlabel('Estimated Probability Range')
            ax.set_ylabel('Bet Magnitude (Absolute)')
            ax.grid(True, alpha=0.3)

    # Confidence vs bet magnitude with probability color coding
    ax = axes[1, 1]
    if ('confidence' in market_metrics and 'odds' in market_metrics and
        len(market_metrics['confidence']) == len(bets) == len(market_metrics['odds'])):
        bet_magnitudes = [abs(bet) for bet in bets]
        scatter = ax.scatter(market_metrics['confidence'], bet_magnitudes,
                           c=market_metrics['odds'], cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Bet Magnitude (Absolute)')
        ax.set_title('Confidence vs Bet Magnitude\n(Color = Estimated Probability)')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Estimated Probability')

    plt.tight_layout()
    plt.savefig(output_dir / 'prob_vs_bet.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_market_consistency_analysis(market_details: Dict[str, List[Dict[str, Any]]], model_name: str, output_dir: Path):
    """Create analysis showing consistency of decisions across multiple runs for the same markets."""
    market_ids = list(market_details.keys())

    if len(market_ids) == 0:
        return

    # Only analyze markets that appear in multiple runs
    multi_run_markets = {mid: data for mid, data in market_details.items() if len(data) > 1}

    if len(multi_run_markets) == 0:
        return

    # Limit to most frequently appearing markets
    market_ids_sorted = sorted(multi_run_markets.keys(), key=lambda x: len(multi_run_markets[x]), reverse=True)
    top_markets = market_ids_sorted[:min(6, len(market_ids_sorted))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Market Decision Consistency Across Runs: {model_name}', fontsize=16, fontweight='bold')

    # Probability consistency by market
    ax = axes[0, 0]
    prob_consistency = []
    market_names = []
    for market_id in top_markets:
        market_data = multi_run_markets[market_id]
        probs = [d['odds'] for d in market_data]
        if len(probs) > 1:
            cv = np.std(probs) / np.mean(probs) if np.mean(probs) != 0 else 0
            prob_consistency.append(cv)
            # Use the actual market question for the label, truncated for readability
            market_question = market_data[0]['market_question']
            truncated_question = market_question[:25] + '...' if len(market_question) > 25 else market_question
            market_names.append(truncated_question)

    if prob_consistency:
        bars = ax.bar(range(len(prob_consistency)), prob_consistency, alpha=0.7, color='skyblue')
        ax.set_title('Probability Estimate Consistency')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_xlabel('Market')
        ax.set_xticks(range(len(market_names)))
        ax.set_xticklabels(market_names, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, prob_consistency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Bet direction consistency
    ax = axes[0, 1]
    direction_consistency = []
    for market_id in top_markets:
        market_data = multi_run_markets[market_id]
        bets = [d['bet'] for d in market_data]
        if len(bets) > 1:
            signs = [1 if bet > 0 else -1 if bet < 0 else 0 for bet in bets]
            # Calculate the proportion of the most common sign
            from collections import Counter
            sign_counts = Counter(signs)
            most_common_count = sign_counts.most_common(1)[0][1]
            consistency = most_common_count / len(signs)
            direction_consistency.append(consistency)

    if direction_consistency:
        bars = ax.bar(range(len(direction_consistency)), direction_consistency, alpha=0.7, color='lightgreen')
        ax.set_title('Bet Direction Consistency')
        ax.set_ylabel('Proportion of Consistent Direction')
        ax.set_xlabel('Market')
        ax.set_xticks(range(len(market_names)))
        ax.set_xticklabels(market_names, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, direction_consistency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Confidence consistency
    ax = axes[0, 2]
    conf_consistency = []
    for market_id in top_markets:
        market_data = multi_run_markets[market_id]
        confidences = [d['confidence'] for d in market_data]
        if len(confidences) > 1:
            cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) != 0 else 0
            conf_consistency.append(cv)

    if conf_consistency:
        bars = ax.bar(range(len(conf_consistency)), conf_consistency, alpha=0.7, color='lightcoral')
        ax.set_title('Confidence Level Consistency')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_xlabel('Market')
        ax.set_xticks(range(len(market_names)))
        ax.set_xticklabels(market_names, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, conf_consistency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Individual market decision patterns (show top 3 markets)
    for plot_idx in range(3):
        ax = axes[1, plot_idx]
        if plot_idx < len(top_markets):
            market_id = top_markets[plot_idx]
            market_data = multi_run_markets[market_id]

            runs = list(range(len(market_data)))
            probs = [d['odds'] for d in market_data]
            bets = [d['bet'] for d in market_data]

            # Create dual-axis plot
            ax2 = ax.twinx()

            line1 = ax.plot(runs, probs, 'o-', color='blue', label='Probability', linewidth=2, markersize=6)
            line2 = ax2.plot(runs, bets, 's-', color='red', label='Bet Amount', linewidth=2, markersize=6)

            ax.set_xlabel('Run Number')
            ax.set_ylabel('Estimated Probability', color='blue')
            ax2.set_ylabel('Bet Amount', color='red')

            # Use actual market question for title, truncated
            market_question = market_data[0]['market_question']
            truncated_question = market_question[:30] + '...' if len(market_question) > 30 else market_question
            ax.set_title(f'{truncated_question}\nDecision Pattern', fontsize=10)

            # Add horizontal line at zero for bets
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'market_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_within_event_structure_analysis(market_by_event_metrics: Dict[str, Dict[str, List[float]]], market_details: Dict[str, List[Dict[str, Any]]], model_name: str, output_dir: Path):
    """Create analysis of how markets within the same event relate to each other."""
    event_ids = list(market_by_event_metrics.keys())

    if len(event_ids) == 0:
        return

    # Focus on events with multiple markets
    multi_market_events = {eid: data for eid, data in market_by_event_metrics.items()
                          if len(data.get('odds', [])) > 1}

    # Create a mapping of event_id to event_title from market_details
    event_titles = {}
    for market_data_list in market_details.values():
        for market_data in market_data_list:
            event_id = market_data['event_id']
            event_title = market_data['event_title']
            if event_id not in event_titles:
                event_titles[event_id] = event_title

    if len(multi_market_events) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Within-Event Market Structure: {model_name}', fontsize=16, fontweight='bold')

    # Probability spread within events
    ax = axes[0, 0]
    prob_spreads = []
    event_labels = []
    for event_id, data in multi_market_events.items():
        if 'odds' in data and len(data['odds']) > 1:
            spread = np.max(data['odds']) - np.min(data['odds'])
            prob_spreads.append(spread)
            # Use actual event title, truncated for readability
            event_title = event_titles.get(event_id, f'Event {event_id}')
            truncated_title = event_title[:20] + '...' if len(event_title) > 20 else event_title
            event_labels.append(truncated_title)

    if prob_spreads:
        bars = ax.bar(range(len(prob_spreads)), prob_spreads, alpha=0.7, color='lightblue')
        ax.set_title('Probability Spread Within Events')
        ax.set_ylabel('Max - Min Probability')
        ax.set_xlabel('Event')
        ax.set_xticks(range(len(event_labels)))
        ax.set_xticklabels(event_labels, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, prob_spreads):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Capital allocation balance within events
    ax = axes[0, 1]
    allocation_imbalances = []
    allocation_event_labels = []
    for event_id, data in multi_market_events.items():
        if 'bets' in data and len(data['bets']) > 1:
            bets = np.array(data['bets'])
            total_long = np.sum(bets[bets > 0])
            total_short = abs(np.sum(bets[bets < 0]))
            total_exposure = total_long + total_short
            if total_exposure > 0:
                imbalance = abs(total_long - total_short) / total_exposure
                allocation_imbalances.append(imbalance)
                # Use actual event title, truncated for readability
                event_title = event_titles.get(event_id, f'Event {event_id}')
                truncated_title = event_title[:15] + '...' if len(event_title) > 15 else event_title
                allocation_event_labels.append(truncated_title)

    if allocation_imbalances:
        bars = ax.bar(range(len(allocation_imbalances)), allocation_imbalances, alpha=0.7, color='orange')
        ax.set_title('Long/Short Imbalance Within Events')
        ax.set_ylabel('Allocation Imbalance Ratio')
        ax.set_xlabel('Event')
        ax.set_xticks(range(len(allocation_imbalances)))
        ax.set_xticklabels(allocation_event_labels, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, allocation_imbalances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Market count distribution
    ax = axes[1, 0]
    market_counts = [len(data.get('odds', [])) for data in multi_market_events.values()]
    if market_counts:
        unique_counts = sorted(set(market_counts))
        count_frequencies = [market_counts.count(c) for c in unique_counts]
        bars = ax.bar(unique_counts, count_frequencies, alpha=0.7, color='lightgreen', width=0.6)
        ax.set_title('Markets per Event Distribution')
        ax.set_xlabel('Number of Markets')
        ax.set_ylabel('Number of Events')
        ax.set_xticks(unique_counts)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, count_frequencies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val}', ha='center', va='bottom', fontsize=10)

    # Confidence correlation with probability diversity
    ax = axes[1, 1]
    prob_diversity = []
    avg_confidence = []

    for event_id, data in multi_market_events.items():
        if 'odds' in data and 'confidence' in data and len(data['odds']) > 1:
            # Use coefficient of variation as measure of probability diversity
            prob_cv = np.std(data['odds']) / np.mean(data['odds']) if np.mean(data['odds']) != 0 else 0
            avg_conf = np.mean(data['confidence'])
            prob_diversity.append(prob_cv)
            avg_confidence.append(avg_conf)

    if prob_diversity and avg_confidence:
        ax.scatter(prob_diversity, avg_confidence, alpha=0.6, s=50, color='purple')
        ax.set_xlabel('Probability Diversity (CV)')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Probability Diversity vs Confidence')
        ax.grid(True, alpha=0.3)

        # Add trend line if we have enough points
        if len(prob_diversity) > 2:
            z = np.polyfit(prob_diversity, avg_confidence, 1)
            p = np.poly1d(z)
            ax.plot(prob_diversity, p(prob_diversity), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'within_event_structure.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_confidence_vs_bet_analysis(market_metrics: Dict[str, List[float]], model_name: str, output_dir: Path):
    """Create detailed analysis of confidence patterns and their relationship to betting behavior."""
    if 'confidence' not in market_metrics or not market_metrics['confidence']:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Confidence vs Betting Behavior: {model_name}', fontsize=16, fontweight='bold')

    confidence_vals = market_metrics['confidence']

    # Confidence distribution
    ax = axes[0, 0]
    ax.hist(confidence_vals, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
    ax.axvline(np.mean(confidence_vals), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(confidence_vals):.2f}')
    ax.axvline(np.median(confidence_vals), color='orange', linestyle='--', linewidth=2,
              label=f'Median: {np.median(confidence_vals):.2f}')
    ax.set_title('Confidence Distribution')
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Confidence vs Bet Magnitude
    ax = axes[0, 1]
    if 'bets' in market_metrics and len(market_metrics['bets']) == len(confidence_vals):
        bet_magnitudes = [abs(bet) for bet in market_metrics['bets']]
        ax.scatter(confidence_vals, bet_magnitudes, alpha=0.6, s=30, color='blue')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Bet Magnitude')
        ax.set_title('Confidence vs Bet Magnitude')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = np.corrcoef(confidence_vals, bet_magnitudes)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Confidence bins vs Bet Direction
    ax = axes[1, 0]
    if 'bets' in market_metrics and len(market_metrics['bets']) == len(confidence_vals):
        # Create confidence bins
        conf_bins = np.linspace(min(confidence_vals), max(confidence_vals), 6)
        bin_labels = [f'{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}' for i in range(len(conf_bins)-1)]

        long_counts = [0] * (len(conf_bins)-1)
        short_counts = [0] * (len(conf_bins)-1)
        no_bet_counts = [0] * (len(conf_bins)-1)

        for conf, bet in zip(confidence_vals, market_metrics['bets']):
            bin_idx = min(int((conf - min(confidence_vals)) / (max(confidence_vals) - min(confidence_vals)) * 5), 4)
            if bet > 0:
                long_counts[bin_idx] += 1
            elif bet < 0:
                short_counts[bin_idx] += 1
            else:
                no_bet_counts[bin_idx] += 1

        x = np.arange(len(bin_labels))
        width = 0.25

        ax.bar(x - width, long_counts, width, label='Long Positions', color='green', alpha=0.7)
        ax.bar(x, short_counts, width, label='Short Positions', color='red', alpha=0.7)
        ax.bar(x + width, no_bet_counts, width, label='No Position', color='gray', alpha=0.7)

        ax.set_title('Position Type by Confidence Level')
        ax.set_xlabel('Confidence Range')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Confidence vs Estimated Probability
    ax = axes[1, 1]
    if 'odds' in market_metrics and len(market_metrics['odds']) == len(confidence_vals):
        # Color code by bet direction
        colors = ['red' if bet < 0 else 'green' if bet > 0 else 'gray'
                 for bet in market_metrics['bets']] if 'bets' in market_metrics else ['blue'] * len(confidence_vals)

        ax.scatter(market_metrics['odds'], confidence_vals, c=colors, alpha=0.6, s=30)
        ax.set_xlabel('Estimated Probability')
        ax.set_ylabel('Confidence')
        ax.set_title('Probability vs Confidence\n(Color = Position Type)')
        ax.grid(True, alpha=0.3)

        # Add legend for colors
        import matplotlib.patches as mpatches
        green_patch = mpatches.Patch(color='green', label='Long')
        red_patch = mpatches.Patch(color='red', label='Short')
        gray_patch = mpatches.Patch(color='gray', label='No Position')
        ax.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right')

        # Add correlation coefficient
        correlation = np.corrcoef(market_metrics['odds'], confidence_vals)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_vs_bet.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results_data: List[ModelInvestmentDecisions],
                market_metrics: Dict[str, List[float]],
                event_metrics: Dict[str, List[float]],
                market_details: Dict[str, List[Dict[str, Any]]],
                market_stats: Dict[str, Dict[str, float]],
                event_stats: Dict[str, Dict[str, float]],
                output_dir: Path = None):
    """Save all results and analysis to files."""
    if output_dir is None:
        output_dir = Path("llm_distribution_analysis")
    output_dir.mkdir(exist_ok=True)

    # Save raw results
    raw_results = []
    for result in results_data:
        raw_results.append(result.model_dump())

    with open(output_dir / 'raw_results.json', 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)

    # Save market-level metrics and statistics
    with open(output_dir / 'market_metrics.json', 'w') as f:
        json.dump(market_metrics, f, indent=2, default=str)

    with open(output_dir / 'market_statistics.json', 'w') as f:
        json.dump(market_stats, f, indent=2, default=str)

    # Save event-level metrics and statistics
    with open(output_dir / 'event_metrics.json', 'w') as f:
        json.dump(event_metrics, f, indent=2, default=str)

    with open(output_dir / 'event_statistics.json', 'w') as f:
        json.dump(event_stats, f, indent=2, default=str)

    # Save market details
    with open(output_dir / 'market_details.json', 'w') as f:
        json.dump(market_details, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")


def create_summary_report(market_stats: Dict[str, Dict[str, float]],
                         event_stats: Dict[str, Dict[str, float]],
                         model_info: ModelInfo,
                         num_runs: int, num_events: int,
                         num_market_decisions: int, num_unique_markets: int,
                         output_dir: Path):
    """Create a human-readable summary report."""
    report_lines = [
        f"LLM Distribution Analysis Report",
        f"=" * 50,
        f"",
        f"Model: {model_info.model_pretty_name}",
        f"Provider: {model_info.company_pretty_name}",
        f"Number of runs: {num_runs}",
        f"Number of events per run: {num_events}",
        f"Total market decisions analyzed: {num_market_decisions}",
        f"Unique markets: {num_unique_markets}",
        f"",
        f"MARKET-LEVEL ANALYSIS:",
        f"=" * 30,
    ]

    # Add market-level statistics
    for metric_name, metric_stats in market_stats.items():
        if metric_name in ['odds', 'bets', 'confidence']:
            display_name = 'Predicted Probability' if metric_name == 'odds' else metric_name.title()
            report_lines.extend([
                f"",
                f"{display_name}:",
                f"  Mean: {metric_stats['mean']:.3f}",
                f"  Std Dev: {metric_stats['std']:.3f}",
                f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]",
                f"  Median: {metric_stats['median']:.3f}",
            ])

    # Add market-level consistency analysis
    report_lines.extend([
        f"",
        f"Market-Level Consistency Analysis:",
        f"-" * 30,
    ])

    for metric_name in ['odds', 'bets', 'confidence']:
        if metric_name in market_stats:
            cv = market_stats[metric_name]['std'] / abs(market_stats[metric_name]['mean']) if market_stats[metric_name]['mean'] != 0 else float('inf')
            consistency = "High" if cv < 0.1 else "Medium" if cv < 0.3 else "Low"
            display_name = 'Predicted Probability' if metric_name == 'odds' else metric_name.title()
            report_lines.append(f"{display_name} consistency: {consistency} (CV: {cv:.3f})")

    # Add event-level analysis
    report_lines.extend([
        f"",
        f"EVENT-LEVEL ANALYSIS:",
        f"=" * 30,
    ])

    for metric_name, metric_stats in event_stats.items():
        if metric_name in ['total_allocated', 'unallocated_capital', 'num_markets_per_event']:
            display_name = metric_name.replace('_', ' ').title()
            report_lines.extend([
                f"",
                f"{display_name}:",
                f"  Mean: {metric_stats['mean']:.3f}",
                f"  Std Dev: {metric_stats['std']:.3f}",
                f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]",
                f"  Median: {metric_stats['median']:.3f}",
            ])

    # Key insights
    report_lines.extend([
        f"",
        f"KEY INSIGHTS:",
        f"=" * 20,
    ])

    if 'num_markets_per_event' in event_stats:
        avg_markets = event_stats['num_markets_per_event']['mean']
        report_lines.append(f"• Average markets per event: {avg_markets:.1f}")

    if 'total_allocated' in event_stats and 'unallocated_capital' in event_stats:
        avg_allocated = event_stats['total_allocated']['mean']
        avg_unallocated = event_stats['unallocated_capital']['mean']
        report_lines.extend([
            f"• Average capital allocation: {avg_allocated:.1%}",
            f"• Average unallocated capital: {avg_unallocated:.1%}"
        ])

    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main function to run the distribution analysis."""

    # Example model configuration - replace with your desired model
    model_info = ModelInfo(
            model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
            model_pretty_name="Qwen3 235B",
            inference_provider="fireworks-ai",
            company_pretty_name="Qwen",
            open_weights=True,
        )

    num_runs = 10
    num_events = 2

    # Run experiments
    logger.info("Starting distribution analysis...")
    results = run_multiple_experiments(model_info, num_runs, num_events)

    if not results:
        logger.error("No results obtained. Exiting.")
        return

    # Extract metrics and calculate statistics
    market_metrics, event_metrics, market_by_event_metrics, market_details = extract_decision_metrics(results)
    market_stats = calculate_statistics(market_metrics)
    event_stats = calculate_statistics(event_metrics)

    # Create output directory
    output_dir = Path("llm_distribution_analysis")
    output_dir.mkdir(exist_ok=True)

    # Create visualizations
    plt.style.use('default')
    sns.set_palette("husl")

    logger.info("Creating market behavior overview...")
    create_market_behavior_overview(market_metrics, model_info.model_pretty_name, output_dir)

    logger.info("Creating event capital allocation analysis...")
    create_event_capital_allocation(event_metrics, model_info.model_pretty_name, output_dir)

    logger.info("Creating probability vs bet analysis...")
    create_probability_bet_analysis(market_metrics, model_info.model_pretty_name, output_dir)

    logger.info("Creating confidence vs bet analysis...")
    create_confidence_vs_bet_analysis(market_metrics, model_info.model_pretty_name, output_dir)

    logger.info("Creating market consistency analysis...")
    create_market_consistency_analysis(market_details, model_info.model_pretty_name, output_dir)

    logger.info("Creating within-event structure analysis...")
    create_within_event_structure_analysis(market_by_event_metrics, market_details, model_info.model_pretty_name, output_dir)

    # Save all results
    logger.info("Saving results...")
    save_results(results, market_metrics, event_metrics, market_details, market_stats, event_stats, output_dir)

    # Create summary report
    logger.info("Creating summary report...")
    num_market_decisions = len(market_metrics.get('odds', []))
    num_unique_markets = len(market_details)
    create_summary_report(market_stats, event_stats, model_info, num_runs, num_events,
                         num_market_decisions, num_unique_markets, output_dir)

    # Print summary
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    logger.info(f"Generated {len(results)} model runs with {num_market_decisions} total market decisions")
    logger.info(f"Analyzed {num_unique_markets} unique markets across {len(results) * num_events} events")
    logger.info(f"Generated visualizations:")
    logger.info(f"  - market_behavior_overview.png (individual market decision patterns)")
    logger.info(f"  - event_allocation.png (capital allocation across events)")
    logger.info(f"  - prob_vs_bet.png (probability estimation vs betting behavior)")
    logger.info(f"  - confidence_vs_bet.png (confidence patterns and betting)")
    logger.info(f"  - market_consistency.png (decision consistency across runs)")
    logger.info(f"  - within_event_structure.png (market relationships within events)")


if __name__ == "__main__":
    main()