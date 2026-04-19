import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, Database, Gauge, Target } from "lucide-react";
import { api } from "@/lib/api";
import { StatCard } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { Skeleton } from "@/components/ui/skeleton";

interface StatsResponse {
  total_records?: number;
  fatal_rate?: number;
  avg_risk_score?: number;
  model_accuracy?: number;
  hourly_risk?: { hour: number; risk: number }[];
  city_risk?: { city: string; risk: number }[];
  weather_impact?: { weather: string; risk: number }[];
  city_ranking?: { city: string; total: number; fatal: number; risk: number }[];
}

const tooltipStyle = {
  contentStyle: {
    background: "hsl(var(--card))",
    border: "1px solid hsl(var(--border))",
    borderRadius: "0.75rem",
    color: "hsl(var(--foreground))",
    fontSize: "12px",
    boxShadow: "0 8px 24px hsl(222 47% 2% / 0.6)",
  },
  labelStyle: { color: "hsl(var(--muted-foreground))", fontSize: "11px", marginBottom: "4px" },
  cursor: { fill: "hsl(var(--primary) / 0.08)" },
};

const Dashboard = () => {
  const [data, setData] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<StatsResponse>("/stats")
      .then((res) => setData(res.data))
      .catch((err) => setError(err.message ?? "Failed to fetch stats"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-80 rounded-xl" />
        <div className="grid gap-6 lg:grid-cols-2">
          <Skeleton className="h-72 rounded-xl" />
          <Skeleton className="h-72 rounded-xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real-time analytics and insights from accident data across Indian cities.
        </p>
      </div>

      {error && (
        <div className="glass-panel flex items-center gap-3 border-destructive/40 p-4 text-sm">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          <span>API unreachable — make sure your backend is running at <code className="font-mono text-primary">http://localhost:8000</code>.</span>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Records"
          value={data?.total_records?.toLocaleString() ?? "—"}
          icon={Database}
          accent="primary"
          hint="accident events analyzed"
        />
        <StatCard
          label="Fatal Rate"
          value={data?.fatal_rate != null ? `${(data.fatal_rate * 100).toFixed(1)}%` : "—"}
          icon={AlertTriangle}
          accent="destructive"
          hint="of total accidents"
        />
        <StatCard
          label="Avg Risk Score"
          value={data?.avg_risk_score != null ? data.avg_risk_score.toFixed(2) : "—"}
          icon={Gauge}
          accent="warning"
          hint="0 (low) – 1 (high)"
        />
        <StatCard
          label="Model Accuracy"
          value={data?.model_accuracy != null ? `${(data.model_accuracy * 100).toFixed(1)}%` : "—"}
          icon={Target}
          accent="success"
          hint="best-performing model"
        />
      </div>

      <ChartCard title="Hourly Risk Pattern" description="Average accident risk score throughout the day">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data?.hourly_risk ?? []} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
                <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="hour" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <Tooltip {...tooltipStyle} />
            <Line
              type="monotone"
              dataKey="risk"
              stroke="hsl(var(--primary))"
              strokeWidth={2.5}
              dot={{ fill: "hsl(var(--primary))", r: 3 }}
              activeDot={{ r: 6, fill: "hsl(var(--primary-glow))" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="City Risk Ranking" description="Mean risk score per metropolitan area">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data?.city_risk ?? []} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="city" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="risk" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Weather Impact" description="How weather conditions affect risk">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data?.weather_impact ?? []} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="weather" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="risk" fill="hsl(var(--primary-glow))" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <ChartCard title="City Ranking Table" description="Detailed breakdown by city">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                <th className="pb-3 pr-4 font-medium">#</th>
                <th className="pb-3 pr-4 font-medium">City</th>
                <th className="pb-3 pr-4 font-medium text-right">Total Accidents</th>
                <th className="pb-3 pr-4 font-medium text-right">Fatal</th>
                <th className="pb-3 font-medium text-right">Avg Risk</th>
              </tr>
            </thead>
            <tbody>
              {(data?.city_ranking ?? []).map((row, i) => (
                <tr key={row.city} className="border-b border-border/50 transition-colors hover:bg-secondary/40">
                  <td className="py-3 pr-4 font-mono text-muted-foreground">{String(i + 1).padStart(2, "0")}</td>
                  <td className="py-3 pr-4 font-medium">{row.city}</td>
                  <td className="py-3 pr-4 text-right font-mono">{row.total?.toLocaleString()}</td>
                  <td className="py-3 pr-4 text-right font-mono text-destructive">{row.fatal?.toLocaleString()}</td>
                  <td className="py-3 text-right">
                    <span className="inline-flex items-center rounded-md bg-primary/10 px-2 py-0.5 font-mono text-xs text-primary">
                      {row.risk?.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
              {(!data?.city_ranking || data.city_ranking.length === 0) && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                    No data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
};

export default Dashboard;
