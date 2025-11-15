"""
Script to create the Athena table for calibration monitoring.
This should be run once to set up the infrastructure.
"""

import sys
import logging
from aws_integration import AWSIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Create the Athena table for calibration monitoring."""
    try:
        logger.info("Initializing AWS integration...")
        aws = AWSIntegration(region_name="us-east-1")
        
        logger.info("Creating Athena table...")
        logger.info("This may take a few moments...")
        
        success = aws.create_table()
        
        if success:
            logger.info("✓ Athena table created successfully!")
            logger.info(f"  Database: {aws.athena_client._client_config.__dict__.get('region_name', 'us-east-1')}")
            logger.info(f"  Table: camera_calibration_monitoring.calibration_results")
            logger.info(f"  Location: s3://camera-calibration-monitoring/iceberg-data/")
            return 0
        else:
            logger.error("✗ Failed to create Athena table")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Error creating table: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

