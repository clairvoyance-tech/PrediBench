#!/usr/bin/env python3
"""Test script for the new baseline betting models."""

from datetime import date, timedelta
from predibench.invest import run_investments_for_specific_date
from predibench.models import MODELS_BY_PROVIDER

def test_baseline_models():
    """Test the new baseline betting models."""
    
    # Get the baseline models we just added
    baseline_models = MODELS_BY_PROVIDER["baseline"]
    
    # Use a recent date for testing
    target_date = date(2024, 12, 1)
    
    print(f"Testing {len(baseline_models)} baseline models...")
    for model in baseline_models:
        print(f"\nTesting model: {model.model_pretty_name} ({model.model_id})")
    
    # Run investments for the baseline models
    results = run_investments_for_specific_date(
        models=baseline_models,
        time_until_ending=timedelta(days=7),  # Look for markets ending within 7 days
        max_n_events=2,  # Test with just 2 events to keep it fast
        target_date=target_date,
        force_rewrite=True,  # Force rewrite to test fresh
    )
    
    print(f"\nTest completed! Got {len(results)} results.")
    
    # Validate results
    assert len(results) == len(baseline_models), f"Expected {len(baseline_models)} results, got {len(results)}"
    
    for i, result in enumerate(results):
        model = baseline_models[i]
        print(f"\nModel: {model.model_pretty_name}")
        print(f"Model ID: {result.model_id}")
        print(f"Target Date: {result.target_date}")
        print(f"Number of events processed: {len(result.event_investment_decisions)}")
        
        # Validate structure
        assert result.model_id == model.model_id
        assert result.target_date == target_date
        assert hasattr(result, "event_investment_decisions")
        
        # Check each event's decisions
        for event_decision in result.event_investment_decisions:
            print(f"  Event: {event_decision.event_title}")
            print(f"    Markets: {len(event_decision.market_investment_decisions)}")
            
            # Validate capital allocation (should sum to ~1.0)
            total_allocated = sum(abs(market.model_decision.bet) 
                                for market in event_decision.market_investment_decisions)
            total_capital = total_allocated + event_decision.unallocated_capital
            print(f"    Total capital allocation: {total_capital:.3f}")
            
            # Should be close to 1.0 (allowing for small floating point errors)
            assert abs(total_capital - 1.0) < 0.001, f"Capital allocation error: {total_capital}"
            
            # Check that rationales exist and are different for different models
            for market in event_decision.market_investment_decisions:
                assert market.model_decision.rationale, "Missing rationale"
                print(f"      Market {market.market_id}: ${market.model_decision.bet:.3f} - {market.model_decision.rationale}")
    
    print("\n✅ All baseline models tested successfully!")
    return results

if __name__ == "__main__":
    test_baseline_models()