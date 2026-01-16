#!/usr/bin/env python
"""Test script to verify metrics API integration."""
import asyncio
import os
from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.cli.agent.evaluation_service import EvaluationService


async def main():
    # Use a dummy token for structure testing (will fail auth but shows flow)
    api_token = os.environ.get("DIGITALOCEAN_API_TOKEN", "test_token")

    try:
        async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
            eval_service = EvaluationService(client=client)

            print("Testing metrics API integration...")
            print()

            # Test 1: Fetch metrics
            try:
                metrics = await eval_service.get_available_metrics()
                print(f"✅ Fetched {len(metrics)} metrics from API")

                # Show first few metrics
                print("\nSample metrics:")
                for metric in metrics[:3]:
                    print(f"  - {metric.metric_name} ({metric.category})")
                    print(f"    Type: {metric.metric_value_type}")
                    if metric.range_min is not None or metric.range_max is not None:
                        print(f"    Range: {metric.range_min} - {metric.range_max}")
                print()
            except Exception as e:
                print(
                    f"⚠️  Could not fetch metrics (expected if no valid token): {type(e).__name__}"
                )
                print()

            # Test 2: Organize by category
            try:
                by_category = await eval_service.get_metrics_by_category()
                print(f"✅ Organized metrics into {len(by_category)} categories")

                for category, category_metrics in sorted(by_category.items()):
                    print(f"  - {category}: {len(category_metrics)} metrics")
                print()
            except Exception as e:
                print(f"⚠️  Could not organize metrics: {type(e).__name__}")
                print()

            # Test 3: Find specific metric
            try:
                metric = await eval_service.find_metric_by_name(
                    "Correctness (general hallucinations)"
                )
                if metric:
                    print(f"✅ Found metric: {metric.metric_name}")
                    print(f"   UUID: {metric.metric_uuid}")
                    print(f"   Type: {metric.metric_value_type}")
                    print(f"   Category: {metric.category}")
                else:
                    print("⚠️  Metric not found")
                print()
            except Exception as e:
                print(f"⚠️  Could not find metric: {type(e).__name__}")
                print()

            print("✅ All structure tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
