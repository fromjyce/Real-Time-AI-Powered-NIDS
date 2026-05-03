import React from "react";
import { Activity, AlertOctagon, Shield, TrendingUp, Zap } from "lucide-react";

export interface MetricsSummary {
  total_alerts_24h: number;
  malicious_count: number;
  suspicious_count: number;
  throughput_pps: number;
  active_flows: number;
  model_accuracy: number;
  drift_detected: boolean;
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
}

function StatCard({ icon, label, value, sub, accent = "text-white" }: StatCardProps) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-slate-400">{label}</span>
      </div>
      <div className={`text-2xl font-bold ${accent}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

interface MetricsPanelProps {
  metrics: MetricsSummary | null;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 animate-pulse">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="bg-slate-800 rounded-xl h-24" />
        ))}
      </div>
    );
  }

  const accuracyPct = (metrics.model_accuracy * 100).toFixed(1);

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      <StatCard
        icon={<AlertOctagon className="w-4 h-4 text-red-400" />}
        label="Alerts (24h)"
        value={metrics.total_alerts_24h.toLocaleString()}
        sub={`${metrics.malicious_count} malicious · ${metrics.suspicious_count} suspicious`}
        accent="text-red-300"
      />
      <StatCard
        icon={<Activity className="w-4 h-4 text-blue-400" />}
        label="Throughput"
        value={`${metrics.throughput_pps.toFixed(0)} pps`}
        sub="packets / second"
      />
      <StatCard
        icon={<Zap className="w-4 h-4 text-purple-400" />}
        label="Active Flows"
        value={metrics.active_flows.toLocaleString()}
        sub="tracked connections"
      />
      <StatCard
        icon={<Shield className="w-4 h-4 text-green-400" />}
        label="Model Accuracy"
        value={`${accuracyPct}%`}
        sub="online learner"
        accent={parseFloat(accuracyPct) > 90 ? "text-green-300" : "text-yellow-300"}
      />
      <StatCard
        icon={<TrendingUp className="w-4 h-4 text-yellow-400" />}
        label="Concept Drift"
        value={metrics.drift_detected ? "Detected" : "Stable"}
        sub={metrics.drift_detected ? "retraining queued" : "model is stable"}
        accent={metrics.drift_detected ? "text-yellow-300" : "text-green-300"}
      />
      <StatCard
        icon={<Activity className="w-4 h-4 text-cyan-400" />}
        label="Malicious Rate"
        value={
          metrics.total_alerts_24h > 0
            ? `${((metrics.malicious_count / metrics.total_alerts_24h) * 100).toFixed(1)}%`
            : "0%"
        }
        sub="of all alerts"
        accent="text-cyan-300"
      />
    </div>
  );
}
