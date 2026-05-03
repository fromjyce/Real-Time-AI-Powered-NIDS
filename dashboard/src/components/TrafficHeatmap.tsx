import React, { useMemo } from "react";
import { Alert } from "./LiveAlerts";

interface HeatmapCell {
  protocol: string;
  port: string;
  count: number;
  maxScore: number;
}

function buildHeatmapData(alerts: Alert[]): HeatmapCell[] {
  const map = new Map<string, { count: number; maxScore: number }>();

  for (const alert of alerts) {
    const port = alert.dst_port ? String(alert.dst_port) : "other";
    const key = `${alert.protocol}::${port}`;
    const existing = map.get(key) ?? { count: 0, maxScore: 0 };
    map.set(key, {
      count: existing.count + 1,
      maxScore: Math.max(existing.maxScore, alert.fusion_score),
    });
  }

  return Array.from(map.entries())
    .map(([key, v]) => {
      const [protocol, port] = key.split("::");
      return { protocol, port, ...v };
    })
    .sort((a, b) => b.count - a.count)
    .slice(0, 40);
}

function scoreToColor(score: number, count: number): string {
  const intensity = Math.min(score * 0.7 + (count / 100) * 0.3, 1);
  if (intensity > 0.8) return "bg-red-600 text-white";
  if (intensity > 0.6) return "bg-orange-500 text-white";
  if (intensity > 0.4) return "bg-yellow-500 text-black";
  if (intensity > 0.2) return "bg-yellow-700 text-white";
  return "bg-slate-700 text-slate-300";
}

interface TrafficHeatmapProps {
  alerts: Alert[];
}

export function TrafficHeatmap({ alerts }: TrafficHeatmapProps) {
  const cells = useMemo(() => buildHeatmapData(alerts), [alerts]);

  return (
    <div>
      <h2 className="text-lg font-semibold text-white mb-3">
        Protocol / Port Heatmap
      </h2>
      {cells.length === 0 ? (
        <div className="text-center text-slate-500 py-8 text-sm">
          No data yet
        </div>
      ) : (
        <div className="grid grid-cols-5 sm:grid-cols-8 gap-1.5">
          {cells.map((cell) => (
            <div
              key={`${cell.protocol}-${cell.port}`}
              className={`rounded p-1.5 text-center transition-colors ${scoreToColor(cell.maxScore, cell.count)}`}
              title={`${cell.protocol}:${cell.port} — ${cell.count} alerts, max score ${cell.maxScore.toFixed(2)}`}
            >
              <div className="text-xs font-mono leading-tight">{cell.port}</div>
              <div className="text-xs opacity-70">{cell.protocol}</div>
              <div className="text-xs font-bold">{cell.count}</div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
        <span>Low risk</span>
        <div className="flex gap-0.5">
          {["bg-slate-700", "bg-yellow-700", "bg-yellow-500", "bg-orange-500", "bg-red-600"].map(
            (c) => (
              <div key={c} className={`w-4 h-3 rounded-sm ${c}`} />
            )
          )}
        </div>
        <span>High risk</span>
      </div>
    </div>
  );
}
