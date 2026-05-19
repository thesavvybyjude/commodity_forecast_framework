"""
P5-T5: Email Notifier
Detects price spikes and sends HTML email alerts to relevant subscribers.
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from notifications.subscriber_manager import SubscriberManager

logger = logging.getLogger(__name__)

SPIKE_ALERT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: auto; }
    .alert-box { background: #fff3cd; border: 1px solid #ffc107; padding: 20px;
                 border-radius: 8px; margin: 20px 0; }
    .price-up { color: #dc3545; font-weight: bold; }
    .recommendation { background: #d4edda; padding: 15px; border-radius: 8px;
                      margin-top: 15px; }
    h2 { color: #333; }
</style></head>
<body>
    <h2>Price Spike Alert</h2>
    <div class="alert-box">
        <p><strong>Commodity:</strong> {{ commodity }}</p>
        <p><strong>Market:</strong> {{ market }}</p>
        <p><strong>Current Price:</strong> NGN {{ current_price }}</p>
        <p><strong>Previous Price (7 days ago):</strong> NGN {{ prev_price }}</p>
        <p class="price-up">Change: +{{ change_pct }}%</p>
    </div>
    <div class="recommendation">
        <p><strong>Recommendation:</strong> {{ recommendation }}</p>
    </div>
    <hr>
    <p style="color: #666; font-size: 12px;">
        Nigeria Commodity Price Forecaster — Automated Alert
    </p>
</body>
</html>
"""


class Notifier:
    """Detects price spikes and sends email alerts."""

    def __init__(self):
        self.subscriber_manager = SubscriberManager()
        self.template = Template(SPIKE_ALERT_TEMPLATE)

    def check_and_alert(self, features_df: pd.DataFrame):
        """Check for price spikes and send alerts."""
        logger.info("Checking for price spikes...")
        spikes = self._check_price_spikes(features_df)

        if spikes:
            logger.info("Detected %d price spike(s)", len(spikes))
            for spike in spikes:
                self.send_spike_alert(**spike)
        else:
            logger.info("No price spikes detected")

    def _check_price_spikes(self, df: pd.DataFrame) -> list:
        """Detect price spikes exceeding PRICE_SPIKE_PCT threshold."""
        spikes = []

        if 'price_ngn' not in df.columns:
            return spikes

        group_cols = [c for c in ['commodity', 'market'] if c in df.columns]
        if not group_cols:
            return spikes

        for name, group in df.groupby(group_cols):
            if len(group) < 8:
                continue

            if 'date' in group.columns:
                group = group.sort_values('date')

            today_price = group['price_ngn'].iloc[-1]
            week_ago_price = group['price_ngn'].iloc[-7]

            if week_ago_price == 0:
                continue

            change = (today_price - week_ago_price) / week_ago_price * 100

            if change >= config.PRICE_SPIKE_PCT:
                if isinstance(name, tuple):
                    commodity = name[0] if len(name) > 0 else 'unknown'
                    market = name[1] if len(name) > 1 else 'unknown'
                else:
                    commodity = str(name)
                    market = 'unknown'

                recommendation = self._generate_recommendation(commodity, change)

                spikes.append({
                    'commodity': commodity,
                    'market': market,
                    'current_price': f"{today_price:,.2f}",
                    'prev_price': f"{week_ago_price:,.2f}",
                    'change_pct': f"{change:.1f}",
                    'recommendation': recommendation,
                })

        return spikes

    def _generate_recommendation(self, commodity: str, change: float) -> str:
        """Generate a recommendation based on the spike magnitude."""
        if change >= 30:
            return (
                f"CRITICAL: {commodity.title()} prices have surged by {change:.1f}% "
                "in 7 days. Consider immediate procurement action or alternative sourcing."
            )
        elif change >= 20:
            return (
                f"HIGH: {commodity.title()} prices up {change:.1f}% this week. "
                "Monitor closely and consider securing supply at current levels."
            )
        else:
            return (
                f"MODERATE: {commodity.title()} prices increased {change:.1f}% this week. "
                "Price trend warrants attention."
            )

    def send_spike_alert(self, commodity: str, market: str,
                         current_price: str, prev_price: str,
                         change_pct: str, recommendation: str):
        """Render and send spike alert to subscribers."""
        html_body = self.template.render(
            commodity=commodity.title(),
            market=market,
            current_price=current_price,
            prev_price=prev_price,
            change_pct=change_pct,
            recommendation=recommendation,
        )

        subject = f"Price Alert: {commodity.title()} +{change_pct}% in {market}"

        subscribers = self.subscriber_manager.get_subscribers(commodity)
        if not subscribers:
            logger.info("No subscribers for %s — skipping email", commodity)
            return

        for email in subscribers:
            self._send(email, subject, html_body)

    def _send(self, to: str, subject: str, html_body: str):
        """Send an HTML email via Gmail SMTP."""
        if config.EMAIL_SENDER == 'your-email@gmail.com':
            logger.warning("Email not configured — skipping send to %s", to)
            return

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = config.EMAIL_SENDER
            msg['To'] = to
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                server.sendmail(config.EMAIL_SENDER, to, msg.as_string())

            logger.info("Email sent to %s: %s", to, subject)

        except Exception as e:
            logger.error("Failed to send email to %s: %s", to, e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Notifier ready. Called by AgentOrchestrator at end of daily pipeline.")
