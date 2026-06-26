"""
SentinelSite — Analytics Service
All aggregation for the dashboard: heatmap, trend charts, OSHA breakdowns,
zone risk ranking, time-of-day analysis, OSHA PDF export.
Strict site_id isolation on every query.
"""
from __future__ import annotations

import io
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, desc, extract, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    EventStatus, NearMissEvent, OshaCategory,
    SeverityLevel, Site, SiteZone, TrainingSample,
    User, UserRole,
)

log = logging.getLogger(__name__)


class AnalyticsService:

    # ── Heatmap data ──────────────────────────────────────────────────────────

    async def get_heatmap_data(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 30,
        status: EventStatus | None = EventStatus.CONFIRMED,
    ) -> list[dict]:
        """
        GPS coordinates + event metadata for Leaflet heatmap.
        Returns all events with valid GPS in the time window.
        """
        since = datetime.utcnow() - timedelta(days=days)
        filters = [
            NearMissEvent.site_id == site_id,
            NearMissEvent.event_timestamp >= since,
            NearMissEvent.gps_lat.isnot(None),
            NearMissEvent.gps_lon.isnot(None),
        ]
        if status:
            filters.append(NearMissEvent.status == status)

        result = await db.execute(
            select(
                NearMissEvent.id,
                NearMissEvent.gps_lat,
                NearMissEvent.gps_lon,
                NearMissEvent.event_timestamp,
                NearMissEvent.status,
                NearMissEvent.yamnet_class,
                NearMissEvent.anomaly_score,
                NearMissEvent.osha_category,
                NearMissEvent.severity,
            )
            .where(and_(*filters))
            .order_by(desc(NearMissEvent.event_timestamp))
        )
        rows = result.all()

        return [
            {
                "event_id": str(r.id),
                "lat": r.gps_lat,
                "lon": r.gps_lon,
                "timestamp": r.event_timestamp.isoformat(),
                "status": r.status.value,
                "yamnet_class": r.yamnet_class,
                "anomaly_score": r.anomaly_score,
                "osha_category": r.osha_category.value if r.osha_category else None,
                "severity": r.severity.value if r.severity else None,
                # Weight for heatmap intensity
                "weight": self._severity_weight(r.severity, r.anomaly_score),
            }
            for r in rows
        ]

    @staticmethod
    def _severity_weight(severity: SeverityLevel | None, anomaly_score: float | None) -> float:
        base = {
            SeverityLevel.HIGH: 3.0,
            SeverityLevel.MEDIUM: 2.0,
            SeverityLevel.LOW: 1.0,
        }.get(severity, 1.0)
        score_factor = anomaly_score or 0.5
        return round(base * score_factor, 3)

    # ── Event trend ───────────────────────────────────────────────────────────

    async def get_event_trend(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 30,
        granularity: str = "day",  # "day" | "week" | "hour"
    ) -> list[dict]:
        """
        Time-series of event counts.
        Used by TrendChart.tsx (Recharts line chart).
        """
        since = datetime.utcnow() - timedelta(days=days)

        if granularity == "hour":
            trunc_fn = func.date_trunc("hour", NearMissEvent.event_timestamp)
        elif granularity == "week":
            trunc_fn = func.date_trunc("week", NearMissEvent.event_timestamp)
        else:
            trunc_fn = func.date_trunc("day", NearMissEvent.event_timestamp)

        result = await db.execute(
            select(
                trunc_fn.label("period"),
                func.count().label("total"),
                func.sum(
                    case((NearMissEvent.status == EventStatus.CONFIRMED, 1), else_=0)
                ).label("confirmed"),
                func.sum(
                    case((NearMissEvent.status == EventStatus.DISMISSED, 1), else_=0)
                ).label("dismissed"),
                func.sum(
                    case((NearMissEvent.status == EventStatus.PENDING, 1), else_=0)
                ).label("pending"),
            )
            .where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.event_timestamp >= since,
            )
            .group_by(text("period"))
            .order_by(text("period"))
        )

        return [
            {
                "period": row.period.isoformat() if row.period else None,
                "total": row.total,
                "confirmed": int(row.confirmed or 0),
                "dismissed": int(row.dismissed or 0),
                "pending": int(row.pending or 0),
            }
            for row in result.all()
        ]

    # ── OSHA breakdown ────────────────────────────────────────────────────────

    async def get_osha_breakdown(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 90,
    ) -> dict[str, Any]:
        """
        OSHA category counts + severity distribution.
        Used by analytics dashboard pie/bar charts.
        """
        since = datetime.utcnow() - timedelta(days=days)
        base_filters = [
            NearMissEvent.site_id == site_id,
            NearMissEvent.status == EventStatus.CONFIRMED,
            NearMissEvent.event_timestamp >= since,
        ]

        # By category
        cat_result = await db.execute(
            select(NearMissEvent.osha_category, func.count().label("n"))
            .where(and_(*base_filters))
            .group_by(NearMissEvent.osha_category)
            .order_by(desc(text("n")))
        )

        # By severity
        sev_result = await db.execute(
            select(NearMissEvent.severity, func.count().label("n"))
            .where(and_(*base_filters))
            .group_by(NearMissEvent.severity)
        )

        # By yamnet_class (top 5 acoustic triggers)
        acoustic_result = await db.execute(
            select(NearMissEvent.yamnet_class, func.count().label("n"))
            .where(and_(*base_filters, NearMissEvent.yamnet_class.isnot(None)))
            .group_by(NearMissEvent.yamnet_class)
            .order_by(desc(text("n")))
            .limit(5)
        )

        return {
            "by_osha_category": [
                {"category": (r.osha_category.value if r.osha_category else "unclassified"), "count": r.n}
                for r in cat_result.all()
            ],
            "by_severity": [
                {"severity": (r.severity.value if r.severity else "unclassified"), "count": r.n}
                for r in sev_result.all()
            ],
            "top_acoustic_triggers": [
                {"yamnet_class": r.yamnet_class, "count": r.n}
                for r in acoustic_result.all()
            ],
            "period_days": days,
        }

    # ── Zone risk ranking ─────────────────────────────────────────────────────

    async def get_zone_risk_ranking(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 30,
    ) -> list[dict]:
        """
        Rank site zones by near-miss density × severity weight.
        Used for supervisor zone-level risk management.
        Events without a zone_id are bucketed as 'unassigned'.
        """
        since = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(
                NearMissEvent.zone_id,
                func.count().label("event_count"),
                func.avg(NearMissEvent.anomaly_score).label("avg_anomaly"),
                func.sum(
                    case(
                        (NearMissEvent.severity == SeverityLevel.HIGH, 3),
                        (NearMissEvent.severity == SeverityLevel.MEDIUM, 2),
                        (NearMissEvent.severity == SeverityLevel.LOW, 1),
                        else_=1,
                    )
                ).label("severity_score"),
            )
            .where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.status == EventStatus.CONFIRMED,
                NearMissEvent.event_timestamp >= since,
            )
            .group_by(NearMissEvent.zone_id)
            .order_by(desc(text("severity_score")))
        )
        rows = result.all()

        # Fetch zone names
        zone_ids = [r.zone_id for r in rows if r.zone_id]
        zone_names: dict[str, str] = {}
        if zone_ids:
            zone_result = await db.execute(
                select(SiteZone.id, SiteZone.name).where(SiteZone.id.in_(zone_ids))
            )
            zone_names = {str(r.id): r.name for r in zone_result.all()}

        return [
            {
                "zone_id": str(r.zone_id) if r.zone_id else None,
                "zone_name": zone_names.get(str(r.zone_id), "Unassigned") if r.zone_id else "Unassigned",
                "event_count": r.event_count,
                "avg_anomaly_score": round(float(r.avg_anomaly or 0), 3),
                "severity_score": int(r.severity_score or 0),
                "risk_level": self._risk_level(int(r.severity_score or 0), r.event_count),
            }
            for r in rows
        ]

    @staticmethod
    def _risk_level(severity_score: int, event_count: int) -> str:
        combined = severity_score + event_count * 0.5
        if combined >= 10:
            return "high"
        elif combined >= 4:
            return "medium"
        return "low"

    # ── Time-of-day analysis ──────────────────────────────────────────────────

    async def get_time_of_day_pattern(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 90,
    ) -> list[dict]:
        """
        Event frequency by hour of day (0–23).
        Identifies high-risk time windows.
        """
        since = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(
                extract("hour", NearMissEvent.event_timestamp).label("hour"),
                func.count().label("count"),
            )
            .where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.status == EventStatus.CONFIRMED,
                NearMissEvent.event_timestamp >= since,
            )
            .group_by(text("hour"))
            .order_by(text("hour"))
        )

        hour_counts = {int(r.hour): r.count for r in result.all()}
        return [
            {"hour": h, "count": hour_counts.get(h, 0)}
            for h in range(24)
        ]

    # ── Summary dashboard card ────────────────────────────────────────────────

    async def get_dashboard_summary(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict:
        """
        Top-level KPIs for the supervisor dashboard overview card.
        """
        now = datetime.utcnow()

        async def count_events(status: EventStatus | None, since: datetime) -> int:
            filters = [
                NearMissEvent.site_id == site_id,
                NearMissEvent.event_timestamp >= since,
            ]
            if status:
                filters.append(NearMissEvent.status == status)
            res = await db.execute(
                select(func.count()).select_from(NearMissEvent).where(and_(*filters))
            )
            return res.scalar() or 0

        # Last 7 days
        since_7d = now - timedelta(days=7)
        # Last 30 days
        since_30d = now - timedelta(days=30)

        total_7d = await count_events(None, since_7d)
        confirmed_7d = await count_events(EventStatus.CONFIRMED, since_7d)
        pending = await count_events(EventStatus.PENDING, datetime.min)
        total_30d = await count_events(None, since_30d)
        confirmed_30d = await count_events(EventStatus.CONFIRMED, since_30d)

        # Pending oldest
        oldest_res = await db.execute(
            select(NearMissEvent.event_timestamp)
            .where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.status == EventStatus.PENDING,
            )
            .order_by(NearMissEvent.event_timestamp.asc())
            .limit(1)
        )
        oldest_pending = oldest_res.scalar_one_or_none()

        false_positive_rate_30d = (
            round((total_30d - confirmed_30d) / max(total_30d, 1), 3)
            if total_30d > 0 else 0.0
        )

        return {
            "pending_review": pending,
            "oldest_pending_at": oldest_pending.isoformat() if oldest_pending else None,
            "last_7_days": {
                "total": total_7d,
                "confirmed": confirmed_7d,
                "false_positive_rate": round((total_7d - confirmed_7d) / max(total_7d, 1), 3),
            },
            "last_30_days": {
                "total": total_30d,
                "confirmed": confirmed_30d,
                "false_positive_rate": false_positive_rate_30d,
            },
            "generated_at": now.isoformat(),
        }

    # ── OSHA PDF export ───────────────────────────────────────────────────────

    async def export_osha_pdf(
        self,
        db: AsyncSession,
        site_id: str,
        event_id: str,
    ) -> bytes:
        """
        Generate OSHA-style incident report PDF for a single confirmed event.
        Uses reportlab (lightweight). Returns PDF bytes.
        """
        # Fetch event
        result = await db.execute(
            select(NearMissEvent).where(
                NearMissEvent.id == event_id,
                NearMissEvent.site_id == site_id,
            )
        )
        event = result.scalar_one_or_none()
        if not event:
            raise ValueError(f"Event {event_id} not found")
        if event.status != EventStatus.CONFIRMED:
            raise ValueError("Only confirmed events can be exported as OSHA reports")

        # Fetch site
        site_res = await db.execute(select(Site).where(Site.id == site_id))
        site = site_res.scalar_one_or_none()

        try:
            return self._generate_pdf(event, site)
        except ImportError:
            log.warning("reportlab not installed — returning plain text fallback")
            return self._generate_text_fallback(event, site).encode()

    def _generate_pdf(self, event: NearMissEvent, site: Site | None) -> bytes:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
        styles = getSampleStyleSheet()
        story = []

        # Header
        story.append(Paragraph("NEAR-MISS INCIDENT REPORT", styles["Title"]))
        story.append(Paragraph("SentinelSite Safety Platform", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * inch))

        # Site info
        story.append(Paragraph(f"Site: {site.name if site else 'Unknown'}", styles["Normal"]))
        story.append(Paragraph(f"Location: {site.location if site else 'N/A'}", styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

        # Incident details table
        data = [
            ["Field", "Value"],
            ["Incident ID", str(event.id)],
            ["Date/Time", event.event_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["GPS Coordinates", f"{event.gps_lat}, {event.gps_lon}" if event.gps_lat else "N/A"],
            ["OSHA Category", event.osha_category.value.replace("_", " ").title() if event.osha_category else "N/A"],
            ["Severity", event.severity.value.upper() if event.severity else "N/A"],
            ["Detection Method", f"Acoustic ({event.yamnet_class or 'N/A'}) + IMU"],
            ["Acoustic Score", f"{event.anomaly_score:.3f}" if event.anomaly_score else "N/A"],
            ["YAMNet Class", event.yamnet_class or "N/A"],
            ["Visual Classification", event.visual_class or "N/A"],
            ["Reviewed By", str(event.reviewed_by) if event.reviewed_by else "N/A"],
            ["Review Notes", event.review_notes or "N/A"],
        ]

        if event.frame_description:
            data.append(["Scene Description", event.frame_description[:200] + "..."])

        table = Table(data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        # Footer
        story.append(Paragraph(
            f"Generated by SentinelSite on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            styles["Italic"]
        ))
        story.append(Paragraph(
            "This report was automatically generated from passive near-miss detection data.",
            styles["Italic"]
        ))

        doc.build(story)
        return buf.getvalue()

    def _generate_text_fallback(
        self,
        event: NearMissEvent,
        site: Site | None,
    ) -> str:
        lines = [
            "NEAR-MISS INCIDENT REPORT — SentinelSite",
            "=" * 50,
            f"Site: {site.name if site else 'Unknown'}",
            f"Incident ID: {event.id}",
            f"Date/Time: {event.event_timestamp.isoformat()}",
            f"OSHA Category: {event.osha_category.value if event.osha_category else 'N/A'}",
            f"Severity: {event.severity.value if event.severity else 'N/A'}",
            f"GPS: {event.gps_lat}, {event.gps_lon}",
            f"YAMNet Class: {event.yamnet_class}",
            f"Anomaly Score: {event.anomaly_score}",
            f"Visual Class: {event.visual_class}",
            f"Frame Description: {event.frame_description or 'N/A'}",
            f"Review Notes: {event.review_notes or 'N/A'}",
            "=" * 50,
            f"Generated: {datetime.utcnow().isoformat()}",
        ]
        return "\n".join(lines)

    # ── Model accuracy trend (for analytics tab) ──────────────────────────────

    async def get_model_performance_trend(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict:
        """
        Recent model accuracy history for both acoustic + visual models.
        Used by dashboard analytics model performance card.
        """
        from app.core.model_service import model_service
        acoustic_trend = await model_service.get_accuracy_history(db, site_id, "acoustic", last_n=10)
        visual_trend = await model_service.get_accuracy_history(db, site_id, "visual", last_n=10)
        return {
            "acoustic": acoustic_trend,
            "visual": visual_trend,
        }

    # ── Training data coverage ────────────────────────────────────────────────

    async def get_training_coverage(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict:
        """
        How many events have been confirmed vs dismissed vs pending.
        Review rate, false positive rate, training dataset growth over time.
        """
        result = await db.execute(
            select(
                NearMissEvent.status,
                func.count().label("count"),
            )
            .where(NearMissEvent.site_id == site_id)
            .group_by(NearMissEvent.status)
        )
        by_status = {r.status.value: r.count for r in result.all()}

        total = sum(by_status.values())
        confirmed = by_status.get("confirmed", 0)
        dismissed = by_status.get("dismissed", 0)
        pending = by_status.get("pending", 0)

        # Training samples used vs available
        sample_res = await db.execute(
            select(
                TrainingSample.is_used_in_training,
                func.count().label("count"),
            )
            .where(TrainingSample.site_id == site_id)
            .group_by(TrainingSample.is_used_in_training)
        )
        samples = {r.is_used_in_training: r.count for r in sample_res.all()}

        return {
            "total_events": total,
            "confirmed": confirmed,
            "dismissed": dismissed,
            "pending": pending,
            "review_rate": round((confirmed + dismissed) / max(total, 1), 3),
            "confirmation_rate": round(confirmed / max(confirmed + dismissed, 1), 3),
            "false_positive_rate": round(dismissed / max(confirmed + dismissed, 1), 3),
            "training_samples": {
                "used": samples.get(True, 0),
                "available": samples.get(False, 0),
                "total": sum(samples.values()),
            },
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
analytics_service = AnalyticsService()
