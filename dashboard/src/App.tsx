import React, { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { Wifi, WifiOff, RefreshCw } from "lucide-react";

import { useWebSocket, WsMessage } from "./hooks/useWebSocket";
import { LiveAlerts, Alert } from "./components/LiveAlerts";
import { AlertTimeline } from "./components/AlertTimeline";
import { TrafficHeatmap } from "./components/TrafficHeatmap";
import { MetricsPanel, MetricsSummary } from "./components/MetricsPanel";

const API_URL = process.env.REACT_APP_API_URL ?? "http://localhost:8000";
const WS_URL = process.env.REACT_APP_WS_URL ?? "ws://localhost:8000/ws";
const MAX_ALERTS = 500;

function ConnectionBadge({ status }: { status: string }) {
  const isOpen = status === "open";
  return (
    <div
      className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-full ${
        isOpen ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
      }`}
    >
      {isOpen ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
      {isOpen ? "Live" : status}
    </div>
  );
}

export default function App() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const metricsTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const onMessage = useCallback((msg: WsMessage) => {
    if (msg.type === "alert") {
      setAlerts((prev) => {
        const alert = msg.payload as Alert;
        return [alert, ...prev].slice(0, MAX_ALERTS);
      });
    }
  }, []);

  const { status } = useWebSocket(WS_URL, { onMessage });

  const fetchMetrics = useCallback(async () => {
    setMetricsLoading(true);
    try {
      const { data } = await axios.get<MetricsSummary>(
        `${API_URL}/api/v1/metrics/summary`
      );
      setMetrics(data);
    } catch {
      // silently fail — metrics are non-critical
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    metricsTimer.current = setInterval(fetchMetrics, 30_000);
    return () => {
      metricsTimer.current && clearInterval(metricsTimer.current);
    };
  }, [fetchMetrics]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <span className="text-white text-sm font-bold">N</span>
          </div>
          <div>
            <h1 className="text-base font-semibold text-white">
              AI-Powered NIDS
            </h1>
            <p className="text-xs text-slate-400">
              Real-Time Network Intrusion Detection
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchMetrics}
            disabled={metricsLoading}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${metricsLoading ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
          <ConnectionBadge status={status} />
        </div>
      </header>

      <main className="p-6 space-y-6 max-w-screen-2xl mx-auto">
        {/* Metrics row */}
        <section>
          <MetricsPanel metrics={metrics} />
        </section>

        {/* Timeline + Heatmap */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <AlertTimeline alerts={alerts} />
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <TrafficHeatmap alerts={alerts} />
          </div>
        </div>

        {/* Live alerts feed */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[520px]">
          <LiveAlerts alerts={alerts} maxVisible={100} />
        </div>
      </main>
    </div>
  );
}
