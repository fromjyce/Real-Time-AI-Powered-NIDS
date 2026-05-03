import React, { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { format } from "date-fns";
import { Alert } from "./LiveAlerts";

interface TimelineBucket {
  time: string;
  malicious: number;
  suspicious: number;
  benign: number;
}

function bucketAlerts(alerts: Alert[], bucketSizeMs: number): TimelineBucket[] {
  if (alerts.length === 0) return [];

  const now = Date.now();
  const windowMs = 30 * 60 * 1000; // last 30 minutes
  const numBuckets = Math.ceil(windowMs / bucketSizeMs);

  const buckets: TimelineBucket[] = Array.from({ length: numBuckets }, (_, i) => {
    const bucketTime = new Date(now - (numBuckets - 1 - i) * bucketSizeMs);
    return {
      time: format(bucketTime, "HH:mm"),
      malicious: 0,
      suspicious: 0,
      benign: 0,
    };
  });

  for (const alert of alerts) {
    const alertMs = alert.timestamp * 1000;
    const offset = now - alertMs;
    if (offset > windowMs) continue;

    const idx = numBuckets - 1 - Math.floor(offset / bucketSizeMs);
    if (idx < 0 || idx >= numBuckets) continue;

    if (alert.severity === "critical" || alert.severity === "high") {
      buckets[idx].malicious += 1;
    } else if (alert.severity === "medium") {
      buckets[idx].suspicious += 1;
    } else {
      buckets[idx].benign += 1;
    }
  }

  return buckets;
}

interface AlertTimelineProps {
  alerts: Alert[];
}

const TOOLTIP_STYLE = {
  backgroundColor: "#1e293b",
  border: "1px solid #334155",
  borderRadius: "6px",
  color: "#f1f5f9",
  fontSize: "12px",
};

export function AlertTimeline({ alerts }: AlertTimelineProps) {
  const data = useMemo(() => bucketAlerts(alerts, 60_000), [alerts]);

  return (
    <div>
      <h2 className="text-lg font-semibold text-white mb-3">Alert Timeline (30m)</h2>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="gradMalicious" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradSuspicious" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradBenign" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend wrapperStyle={{ fontSize: "12px", color: "#94a3b8" }} />
          <Area
            type="monotone"
            dataKey="malicious"
            stroke="#ef4444"
            fill="url(#gradMalicious)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="suspicious"
            stroke="#f59e0b"
            fill="url(#gradSuspicious)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="benign"
            stroke="#22c55e"
            fill="url(#gradBenign)"
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
