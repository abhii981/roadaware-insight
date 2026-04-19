import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { ChartCard } from "@/components/ChartCard";
import { Skeleton } from "@/components/ui/skeleton";

interface ModelMetric {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

const tooltipStyle = {
  contentStyle: {
    background: "hsl(var(--card))",
    border: "1px solid hsl(var(--border))",
    borderRadius: "0.75rem",
    color: "hsl(var(--foreground))",
    fontSize: "12px",
  },
  labelStyle: { color: "hsl(var(--muted-foreground))", fontSize: "11px" },
  cursor: { fill: "hsl(var(--primary) / 0.08)" },
};

const ModelPerformance = () => {
  const [data, setData] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<ModelMetric[] | { models: ModelMetric[] }>("/model-performance")
      .then((res) => {
        const arr = Array.isArray(res.data) ? res.data : (res.data as any).models ?? [];
        setData(arr);
      })
      .catch((err) => setError(err.message ?? "Failed to fetch metrics"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
    );
  }

  const fmt = (v: number) => (v != null ? `${(v).toFixed(1)}%` : "—");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">Model Performance</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Compare classification metrics across all trained models.
        </p>
      </div>

      {error && (
        <div className="glass-panel flex items-center gap-3 border-destructive/40 p-4 text-sm">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          <span>{error}</span>
        </div>
      )}

      <ChartCard title="Metric Comparison" description="Higher is better — accuracy, precision, recall and F1-score">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="model" stroke="hsl(var(--muted-foreground))" fontSize={11} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} domain={[0, 1]} />
            <Tooltip {...tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
            <Bar dataKey="accuracy" name="Accuracy" fill="hsl(217 91% 60%)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="precision" name="Precision" fill="hsl(199 89% 65%)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="recall" name="Recall" fill="hsl(142 71% 45%)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="f1" name="F1 Score" fill="hsl(38 92% 50%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Detailed Metrics" description="Per-model classification report">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
                <th className="pb-3 pr-4 font-medium">Model</th>
                <th className="pb-3 pr-4 font-medium text-right">Accuracy</th>
                <th className="pb-3 pr-4 font-medium text-right">Precision</th>
                <th className="pb-3 pr-4 font-medium text-right">Recall</th>
                <th className="pb-3 font-medium text-right">F1 Score</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <tr key={row.model} className="border-b border-border/50 transition-colors hover:bg-secondary/40">
                  <td className="py-4 pr-4 font-medium">{row.model}</td>
                  <td className="py-4 pr-4 text-right font-mono text-primary">{fmt(row.accuracy)}</td>
                  <td className="py-4 pr-4 text-right font-mono">{fmt(row.precision)}</td>
                  <td className="py-4 pr-4 text-right font-mono">{fmt(row.recall)}</td>
                  <td className="py-4 text-right font-mono">{fmt(row.f1)}</td>
                </tr>
              ))}
              {data.length === 0 && (
                <tr><td colSpan={5} className="py-8 text-center text-muted-foreground">No metrics available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
};

export default ModelPerformance;
