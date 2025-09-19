"""
Model Consistency Analysis Script

This script analyzes the consistency of LLM-based prediction market agents across multiple runs.
It loads cached results from the bucket-prod folder and generates scientific visualizations
showing various aspects of model consistency and behavior patterns.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from predibench.agent.models import ModelInfo, ModelInvestmentDecisions
from predibench.logger_config import get_logger
from predibench.utils import apply_template

logger = get_logger(__name__)


def load_cached_model_runs(
    model_id: str,
    target_date: str = "2025-09-18",
    num_runs: int = 32
) -> List[ModelInvestmentDecisions]:
    """Load cached model runs from bucket-prod folder."""
    base_path = Path("/Users/charlesazam/charloupioupiou/market-bench/bucket-prod/model_results")
    date_path = base_path / target_date

    # Convert model_id to the directory format used in bucket-prod
    safe_model_id = model_id.replace('/', '--')

    results = []
    for run_idx in range(num_runs):
        run_path = date_path / f"{safe_model_id}_run_{run_idx}"
        result_file = run_path / "model_investment_decisions.json"

        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                results.append(ModelInvestmentDecisions(**data))
                logger.info(f"Loaded run {run_idx} for {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load run {run_idx} for {model_id}: {e}")
        else:
            logger.warning(f"Run {run_idx} not found for {model_id}")

    logger.info(f"Successfully loaded {len(results)} runs for {model_id}")
    return results


def extract_consistency_metrics(results: List[ModelInvestmentDecisions]) -> Dict[str, Any]:
    """Extract metrics for consistency analysis."""
    # Market-level data grouped by market_id
    market_data = defaultdict(list)

    # Event-level data grouped by event_id
    event_data = defaultdict(list)

    # Overall run-level statistics
    run_level_stats = []

    for run_idx, result in enumerate(results):
        run_bets = []
        run_probs = []
        run_confidences = []
        run_total_allocated = 0
        run_num_decisions = 0

        for event_decision in result.event_investment_decisions:
            event_id = event_decision.event_id
            event_allocated = 0
            event_bets = []
            event_probs = []
            event_confidences = []

            for market_decision in event_decision.market_investment_decisions:
                market_id = market_decision.market_id
                decision = market_decision.decision

                # Store market-level data for consistency analysis
                market_data[market_id].append({
                    'run_idx': run_idx,
                    'event_id': event_id,
                    'event_title': event_decision.event_title,
                    'market_question': market_decision.market_question,
                    'estimated_probability': decision.estimated_probability,
                    'bet': decision.bet,
                    'confidence': decision.confidence,
                    'rationale': decision.rationale
                })

                # Accumulate for event and run level stats
                event_allocated += abs(decision.bet)
                event_bets.append(decision.bet)
                event_probs.append(decision.estimated_probability)
                event_confidences.append(decision.confidence)

                run_bets.append(decision.bet)
                run_probs.append(decision.estimated_probability)
                run_confidences.append(decision.confidence)
                run_total_allocated += abs(decision.bet)
                run_num_decisions += 1

            # Store event-level data
            event_data[event_id].append({
                'run_idx': run_idx,
                'event_title': event_decision.event_title,
                'total_allocated': event_allocated,
                'unallocated_capital': event_decision.unallocated_capital,
                'num_markets': len(event_decision.market_investment_decisions),
                'avg_probability': np.mean(event_probs) if event_probs else 0,
                'avg_confidence': np.mean(event_confidences) if event_confidences else 0,
                'bet_variance': np.var(event_bets) if event_bets else 0
            })

        # Store run-level statistics
        run_level_stats.append({
            'run_idx': run_idx,
            'total_allocated': run_total_allocated,
            'num_decisions': run_num_decisions,
            'avg_probability': np.mean(run_probs) if run_probs else 0,
            'avg_confidence': np.mean(run_confidences) if run_confidences else 0,
            'avg_bet_magnitude': np.mean([abs(b) for b in run_bets]) if run_bets else 0,
            'bet_variance': np.var(run_bets) if run_bets else 0
        })

    return {
        'market_data': dict(market_data),
        'event_data': dict(event_data),
        'run_level_stats': run_level_stats
    }


def calculate_consistency_scores(data: List[Dict[str, Any]], metric: str) -> Dict[str, float]:
    """Calculate various consistency scores for a given metric."""
    values = [d[metric] for d in data if metric in d]

    if len(values) < 2:
        return {'cv': 0, 'std': 0, 'range': 0, 'iqr': 0}

    mean_val = np.mean(values)
    std_val = np.std(values)

    return {
        'cv': std_val / abs(mean_val) if mean_val != 0 else float('inf'),
        'std': std_val,
        'range': np.max(values) - np.min(values),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'mean': mean_val,
        'median': np.median(values)
    }


def create_market_consistency_heatmap(consistency_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Create a heatmap showing consistency scores across different markets and metrics."""
    market_data = consistency_data['market_data']

    # Filter markets that appear in multiple runs
    multi_run_markets = {mid: data for mid, data in market_data.items()
                        if len(data) >= 5}  # At least 5 runs

    if not multi_run_markets:
        return go.Figure()

    # Calculate consistency scores for each market and metric
    metrics = ['estimated_probability', 'bet', 'confidence']
    market_ids = list(multi_run_markets.keys())[:15]  # Top 15 markets

    consistency_matrix = []
    market_labels = []

    for market_id in market_ids:
        data = multi_run_markets[market_id]
        row = []

        for metric in metrics:
            scores = calculate_consistency_scores(data, metric)
            row.append(scores['cv'])  # Use coefficient of variation

        consistency_matrix.append(row)
        # Use market question as label, truncated
        market_question = data[0]['market_question']
        truncated = market_question[:30] + '...' if len(market_question) > 30 else market_question
        market_labels.append(truncated)

    fig = go.Figure(data=go.Heatmap(
        z=consistency_matrix,
        x=['Probability', 'Bet Amount', 'Confidence'],
        y=market_labels,
        colorscale='RdYlBu_r',
        text=[[f'{val:.3f}' for val in row] for row in consistency_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Coefficient of Variation<br>(Lower = More Consistent)")
    ))

    fig.update_layout(
        title=f'{model_name}: Market Decision Consistency',
        xaxis_title='Decision Metrics',
        yaxis_title='Markets',
        height=max(400, len(market_labels) * 25)
    )

    return fig


def create_run_level_consistency_analysis(consistency_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Create analysis of consistency at the run level."""
    run_stats = consistency_data['run_level_stats']

    if not run_stats:
        return go.Figure()

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Total Allocation per Run', 'Average Probability per Run',
                       'Average Confidence per Run', 'Bet Variance per Run'],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    run_indices = [s['run_idx'] for s in run_stats]

    # Total allocation
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[s['total_allocated'] for s in run_stats],
            mode='lines+markers',
            name='Total Allocated',
            line=dict(color='blue'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Average probability
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[s['avg_probability'] for s in run_stats],
            mode='lines+markers',
            name='Avg Probability',
            line=dict(color='green'),
            marker=dict(size=6)
        ),
        row=1, col=2
    )

    # Average confidence
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[s['avg_confidence'] for s in run_stats],
            mode='lines+markers',
            name='Avg Confidence',
            line=dict(color='red'),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    # Bet variance
    fig.add_trace(
        go.Scatter(
            x=run_indices,
            y=[s['bet_variance'] for s in run_stats],
            mode='lines+markers',
            name='Bet Variance',
            line=dict(color='purple'),
            marker=dict(size=6)
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f'{model_name}: Run-Level Consistency Analysis',
        showlegend=False,
        height=600
    )

    # Update x-axis labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Run Number", row=i, col=j)

    return fig


def create_probability_bet_consistency_scatter(consistency_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Create scatter plot showing consistency relationship between probability and bet decisions."""
    market_data = consistency_data['market_data']

    # Calculate consistency scores for markets with multiple runs
    prob_consistency = []
    bet_consistency = []
    market_names = []

    for market_id, data in market_data.items():
        if len(data) >= 5:  # At least 5 runs
            prob_scores = calculate_consistency_scores(data, 'estimated_probability')
            bet_scores = calculate_consistency_scores(data, 'bet')

            prob_consistency.append(prob_scores['cv'])
            bet_consistency.append(bet_scores['cv'])

            market_question = data[0]['market_question']
            truncated = market_question[:20] + '...' if len(market_question) > 20 else market_question
            market_names.append(truncated)

    if not prob_consistency:
        return go.Figure()

    fig = go.Figure(data=go.Scatter(
        x=prob_consistency,
        y=bet_consistency,
        mode='markers',
        text=market_names,
        hovertemplate='<b>%{text}</b><br>' +
                      'Probability CV: %{x:.3f}<br>' +
                      'Bet CV: %{y:.3f}<extra></extra>',
        marker=dict(
            size=10,
            color=prob_consistency,
            colorscale='Viridis',
            colorbar=dict(title="Probability CV"),
            line=dict(width=1, color='white')
        )
    ))

    # Add diagonal line for reference
    max_val = max(max(prob_consistency), max(bet_consistency))
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.update_layout(
        title=f'{model_name}: Probability vs Bet Consistency',
        xaxis_title='Probability Estimate Consistency (CV)',
        yaxis_title='Bet Decision Consistency (CV)',
        annotations=[
            dict(
                x=max_val*0.7, y=max_val*0.3,
                text="Equal Consistency Line",
                showarrow=False,
                font=dict(color="red")
            )
        ]
    )

    return fig


def create_confidence_patterns_analysis(consistency_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Analyze confidence patterns and their relationship to consistency."""
    market_data = consistency_data['market_data']

    # Collect all confidence values and corresponding consistency metrics
    confidence_ranges = [(1, 3), (4, 6), (7, 10)]
    range_labels = ['Low (1-3)', 'Medium (4-6)', 'High (7-10)']

    prob_cv_by_conf = []
    bet_cv_by_conf = []

    for conf_min, conf_max in confidence_ranges:
        range_prob_cvs = []
        range_bet_cvs = []

        for market_id, data in market_data.items():
            if len(data) >= 5:
                # Filter data by confidence range
                filtered_data = [d for d in data if conf_min <= d['confidence'] <= conf_max]

                if len(filtered_data) >= 3:  # Need at least 3 points for meaningful CV
                    prob_scores = calculate_consistency_scores(filtered_data, 'estimated_probability')
                    bet_scores = calculate_consistency_scores(filtered_data, 'bet')

                    range_prob_cvs.append(prob_scores['cv'])
                    range_bet_cvs.append(bet_scores['cv'])

        prob_cv_by_conf.append(range_prob_cvs)
        bet_cv_by_conf.append(range_bet_cvs)

    # Create box plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Probability Consistency by Confidence', 'Bet Consistency by Confidence']
    )

    colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Probability consistency
    for i, (cvs, label, color) in enumerate(zip(prob_cv_by_conf, range_labels, colors)):
        if cvs:
            fig.add_trace(
                go.Box(y=cvs, name=label, marker_color=color, showlegend=False),
                row=1, col=1
            )

    # Bet consistency
    for i, (cvs, label, color) in enumerate(zip(bet_cv_by_conf, range_labels, colors)):
        if cvs:
            fig.add_trace(
                go.Box(y=cvs, name=label, marker_color=color, showlegend=False),
                row=1, col=2
            )

    fig.update_layout(
        title=f'{model_name}: Consistency by Confidence Level',
        height=400
    )

    fig.update_yaxes(title_text="Coefficient of Variation", row=1, col=1)
    fig.update_yaxes(title_text="Coefficient of Variation", row=1, col=2)
    fig.update_xaxes(title_text="Confidence Range", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Range", row=1, col=2)

    return fig


def create_event_consistency_comparison(consistency_data: Dict[str, Any], model_name: str) -> go.Figure:
    """Compare consistency across different events."""
    event_data = consistency_data['event_data']

    # Calculate consistency metrics for each event
    event_metrics = []

    for event_id, data in event_data.items():
        if len(data) >= 5:  # At least 5 runs
            metrics = {}

            for metric in ['total_allocated', 'avg_probability', 'avg_confidence']:
                scores = calculate_consistency_scores(data, metric)
                metrics[f'{metric}_cv'] = scores['cv']
                metrics[f'{metric}_mean'] = scores['mean']

            metrics['event_title'] = data[0]['event_title']
            metrics['num_runs'] = len(data)
            event_metrics.append(metrics)

    if not event_metrics:
        return go.Figure()

    # Sort by total allocation consistency
    event_metrics.sort(key=lambda x: x['total_allocated_cv'])

    # Take top 10 most consistent events
    top_events = event_metrics[:10]

    event_names = [e['event_title'][:25] + '...' if len(e['event_title']) > 25
                   else e['event_title'] for e in top_events]

    fig = go.Figure()

    # Add bars for different metrics
    metrics_to_plot = [
        ('total_allocated_cv', 'Total Allocation', 'blue'),
        ('avg_probability_cv', 'Avg Probability', 'green'),
        ('avg_confidence_cv', 'Avg Confidence', 'red')
    ]

    for metric, label, color in metrics_to_plot:
        fig.add_trace(go.Bar(
            x=event_names,
            y=[e[metric] for e in top_events],
            name=label,
            marker_color=color,
            opacity=0.7
        ))

    fig.update_layout(
        title=f'{model_name}: Event-Level Consistency Comparison',
        xaxis_title='Events',
        yaxis_title='Coefficient of Variation (Lower = More Consistent)',
        xaxis={'categoryorder': 'array', 'categoryarray': event_names},
        barmode='group',
        height=500
    )

    fig.update_xaxes(tickangle=45)

    return fig


def create_comparison_across_models(gpt_data: Dict[str, Any], qwen_data: Dict[str, Any]) -> go.Figure:
    """Compare consistency metrics between GPT and Qwen models."""

    def get_overall_consistency_stats(data):
        run_stats = data['run_level_stats']
        if not run_stats:
            return {}

        stats = {}
        for metric in ['total_allocated', 'avg_probability', 'avg_confidence', 'bet_variance']:
            values = [s[metric] for s in run_stats]
            if values:
                mean_val = np.mean(values)
                cv = np.std(values) / abs(mean_val) if mean_val != 0 else 0
                stats[metric] = cv

        return stats

    gpt_stats = get_overall_consistency_stats(gpt_data)
    qwen_stats = get_overall_consistency_stats(qwen_data)

    if not gpt_stats or not qwen_stats:
        return go.Figure()

    metrics = list(gpt_stats.keys())
    metric_labels = ['Total Allocation', 'Avg Probability', 'Avg Confidence', 'Bet Variance']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metric_labels,
        y=[gpt_stats[m] for m in metrics],
        name='GPT-OSS 120B',
        marker_color='lightblue',
        opacity=0.8
    ))

    fig.add_trace(go.Bar(
        x=metric_labels,
        y=[qwen_stats[m] for m in metrics],
        name='Qwen3 Coder 480B',
        marker_color='lightcoral',
        opacity=0.8
    ))

    fig.update_layout(
        title='Model Consistency Comparison: GPT vs Qwen',
        xaxis_title='Metrics',
        yaxis_title='Coefficient of Variation (Lower = More Consistent)',
        barmode='group',
        height=500
    )

    return fig


def main():
    """Main function to run the consistency analysis."""
    logger.info("Starting Model Consistency Analysis...")

    # Create output directory
    output_dir = Path("/Users/charlesazam/charloupioupiou/market-bench/analyses/model_consistency")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define models to analyze
    models = [
        {
            'id': 'openai/gpt-oss-120b',
            'name': 'GPT-OSS 120B'
        },
        {
            'id': 'Qwen/Qwen3-Coder-480B-A35B-Instruct',
            'name': 'Qwen3 Coder 480B'
        }
    ]

    model_data = {}

    # Load and analyze each model
    for model in models:
        logger.info(f"Loading data for {model['name']}...")

        # Load cached results
        results = load_cached_model_runs(model['id'])

        if not results:
            logger.warning(f"No results found for {model['name']}")
            continue

        # Extract consistency metrics
        consistency_data = extract_consistency_metrics(results)
        model_data[model['id']] = consistency_data

        # Create individual model analyses
        logger.info(f"Creating consistency analyses for {model['name']}...")

        # Market consistency heatmap
        fig1 = create_market_consistency_heatmap(consistency_data, model['name'])
        if fig1.data:
            apply_template(fig1)
            fig1.update_layout(width=800, height=600)
            fig1.write_html(output_dir / f"{model['name'].replace(' ', '_').lower()}_market_consistency_heatmap.html")

        # Run-level consistency
        fig2 = create_run_level_consistency_analysis(consistency_data, model['name'])
        if fig2.data:
            apply_template(fig2)
            fig2.update_layout(width=1000, height=700)
            fig2.write_html(output_dir / f"{model['name'].replace(' ', '_').lower()}_run_level_consistency.html")

        # Probability vs bet consistency scatter
        fig3 = create_probability_bet_consistency_scatter(consistency_data, model['name'])
        if fig3.data:
            apply_template(fig3)
            fig3.update_layout(width=700, height=600)
            fig3.write_html(output_dir / f"{model['name'].replace(' ', '_').lower()}_prob_bet_consistency.html")

        # Confidence patterns
        fig4 = create_confidence_patterns_analysis(consistency_data, model['name'])
        if fig4.data:
            apply_template(fig4)
            fig4.update_layout(width=800, height=500)
            fig4.write_html(output_dir / f"{model['name'].replace(' ', '_').lower()}_confidence_patterns.html")

        # Event consistency comparison
        fig5 = create_event_consistency_comparison(consistency_data, model['name'])
        if fig5.data:
            apply_template(fig5)
            fig5.update_layout(width=1000, height=600)
            fig5.write_html(output_dir / f"{model['name'].replace(' ', '_').lower()}_event_consistency.html")

    # Create cross-model comparison
    if len(model_data) >= 2:
        logger.info("Creating cross-model comparison...")
        gpt_key = 'openai/gpt-oss-120b'
        qwen_key = 'Qwen/Qwen3-Coder-480B-A35B-Instruct'

        if gpt_key in model_data and qwen_key in model_data:
            fig_comparison = create_comparison_across_models(
                model_data[gpt_key],
                model_data[qwen_key]
            )
            if fig_comparison.data:
                apply_template(fig_comparison)
                fig_comparison.update_layout(width=800, height=600)
                fig_comparison.write_html(output_dir / "model_consistency_comparison.html")

    logger.info(f"Analysis complete! Results saved to {output_dir}")

    # Print summary statistics
    for model in models:
        if model['id'] in model_data:
            data = model_data[model['id']]
            run_stats = data['run_level_stats']

            if run_stats:
                total_allocated_cv = np.std([s['total_allocated'] for s in run_stats]) / np.mean([s['total_allocated'] for s in run_stats])
                prob_cv = np.std([s['avg_probability'] for s in run_stats]) / np.mean([s['avg_probability'] for s in run_stats])
                conf_cv = np.std([s['avg_confidence'] for s in run_stats]) / np.mean([s['avg_confidence'] for s in run_stats])

                logger.info(f"\n{model['name']} Consistency Summary:")
                logger.info(f"  Total Allocation CV: {total_allocated_cv:.3f}")
                logger.info(f"  Average Probability CV: {prob_cv:.3f}")
                logger.info(f"  Average Confidence CV: {conf_cv:.3f}")
                logger.info(f"  Number of runs analyzed: {len(run_stats)}")


if __name__ == "__main__":
    main()