import React, { useEffect, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, Info, ShieldAlert } from "lucide-react";

export interface Alert {
  id: string;
  timestamp: number;
  src_ip: string;
  dst_ip: string;
  dst_port: number | null;
  protocol: string;
  threat_label: string;
  attack_class: string;
  severity: "low" | "medium" | "high" | "critical";
  fusion_score: number;
  matched_rules: string[];
}

interface LiveAlertsProps {
  alerts: Alert[];
  maxVisible?: number;
}

const SEVERITY_CONFIG = {
  critical: {
    bg: "bg-red-950",
    border: "border-red-500",
    badge: "bg-red-500 text-white",
    icon: <ShieldAlert className="w-4 h-4 text-red-400" />,
  },
  high: {
    bg: "bg-orange-950",
    border: "border-orange-500",
    badge: "bg-orange-500 text-white",
    icon: <AlertCircle className="w-4 h-4 text-orange-400" />,
  },
  medium: {
    bg: "bg-yellow-950",
    border: "border-yellow-500",
    badge: "bg-yellow-500 text-black",
    icon: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
  },
  low: {
    bg: "bg-slate-800",
    border: "border-slate-600",
    badge: "bg-slate-500 text-white",
    icon: <Info className="w-4 h-4 text-slate-400" />,
  },
};

function formatTs(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full bg-slate-700 rounded-full h-1.5">
      <div
        className={`h-1.5 rounded-full ${color}`}
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
  );
}

export function LiveAlerts({ alerts, maxVisible = 50 }: LiveAlertsProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = 0;
    }
  }, [alerts, autoScroll]);

  const visible = alerts.slice(0, maxVisible);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white">Live Alerts</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-400">{alerts.length} total</span>
          <label className="flex items-center gap-1 text-xs text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="w-3 h-3"
            />
            Auto-scroll
          </label>
        </div>
      </div>

      <div ref={listRef} className="flex-1 overflow-y-auto space-y-2 pr-1">
        {visible.length === 0 && (
          <div className="text-center text-slate-500 py-12">
            <ShieldAlert className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>No alerts yet. Waiting for traffic…</p>
          </div>
        )}

        {visible.map((alert) => {
          const cfg = SEVERITY_CONFIG[alert.severity] ?? SEVERITY_CONFIG.low;
          return (
            <div
              key={alert.id}
              className={`rounded-lg border p-3 ${cfg.bg} ${cfg.border} transition-all`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  {cfg.icon}
                  <span className="text-sm font-medium text-white">
                    {alert.attack_class}
                  </span>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${cfg.badge}`}>
                    {alert.severity.toUpperCase()}
                  </span>
                </div>
                <span className="text-xs text-slate-400 shrink-0">
                  {formatTs(alert.timestamp)}
                </span>
              </div>

              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-300">
                <span>
                  <span className="text-slate-500">src </span>
                  {alert.src_ip}
                </span>
                <span>
                  <span className="text-slate-500">dst </span>
                  {alert.dst_ip}:{alert.dst_port ?? "—"}
                </span>
                <span>
                  <span className="text-slate-500">proto </span>
                  {alert.protocol}
                </span>
                <span>
                  <span className="text-slate-500">score </span>
                  {alert.fusion_score.toFixed(3)}
                </span>
              </div>

              <div className="mt-2">
                <ScoreBar
                  value={alert.fusion_score}
                  color={
                    alert.severity === "critical"
                      ? "bg-red-500"
                      : alert.severity === "high"
                      ? "bg-orange-500"
                      : alert.severity === "medium"
                      ? "bg-yellow-500"
                      : "bg-slate-500"
                  }
                />
              </div>

              {alert.matched_rules.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {alert.matched_rules.map((r) => (
                    <span
                      key={r}
                      className="text-xs bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded"
                    >
                      {r}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
