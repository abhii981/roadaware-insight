import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
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
  labelStyle: {
    color: "hsl(var(--muted-foreground))",
    fontSize: "11px",
  },
  cursor: {
    fill: "hsl(var(--primary) / 0.08)",
  },
};

const ModelPerformance = () => {
  const [data, setData] = useState<ModelMetric[]>([]);
  const [confusion, setConfusion] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get("/model-performance")
      .then((res) => {
        const arr = Array.isArray(res.data)
          ? res.data
          : res.data.models ?? [];
        setData(arr);
      })
      .catch((err) =>
        setError(err.message ?? "Failed to fetch metrics")
      )
      .finally(() => setLoading(false));

    api.get("/confusion-matrices").then((res) => {
      setConfusion(res.data);
    });
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

  const fmt = (v: number) => (v != null ? `${v.toFixed(1)}%` : "—");

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Model Performance</h1>
        <p className="text-sm text-muted-foreground">
          Compare classification metrics across all trained models.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-3 border rounded bg-red-100 text-red-600">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Chart */}
      <ChartCard title="Metric Comparison">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis domain={[0, 100]} />
            <Tooltip {...tooltipStyle} />
            <Legend />

            <Bar dataKey="accuracy" fill="#3b82f6" />
            <Bar dataKey="precision" fill="#06b6d4" />
            <Bar dataKey="recall" fill="#22c55e" />
            <Bar dataKey="f1" fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Table */}
      <ChartCard title="Detailed Metrics">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left">
                <th>Model</th>
                <th className="text-right">Accuracy</th>
                <th className="text-right">Precision</th>
                <th className="text-right">Recall</th>
                <th className="text-right">F1</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => (
                <tr key={row.model} className="border-b">
                  <td className="py-2">{row.model}</td>
                  <td className="text-right">{fmt(row.accuracy)}</td>
                  <td className="text-right">{fmt(row.precision)}</td>
                  <td className="text-right">{fmt(row.recall)}</td>
                  <td className="text-right">{fmt(row.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {/* Confusion Matrix */}
      {confusion && (
        <div className="mt-10">
          <h2 className="text-xl font-bold mb-6">Confusion Matrix</h2>

          {Object.entries(confusion).map(([model, obj]: any) => (
            <div key={model} className="mb-10">
              <h3 className="font-semibold mb-4">{model}</h3>

              <div className="flex justify-center">
                <div className="grid grid-cols-3 gap-3 bg-gray-800 p-4 rounded-xl">
                  {obj.matrix.map((row: number[], i: number) =>
                    row.map((cell: number, j: number) => {
                      const intensity = Math.min(cell / 2000, 1);

                      return (
                        <div
                          key={`${i}-${j}`}
                          className="w-20 h-20 flex items-center justify-center rounded-lg text-white font-semibold"
                          style={{
                            backgroundColor: `rgba(59,130,246, ${0.3 + intensity})`,
                          }}
                        >
                          {cell}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <p className="text-xs text-center text-gray-400 mt-3">
                Rows = Actual | Columns = Predicted
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelPerformance;