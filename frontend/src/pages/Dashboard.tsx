import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, Database, Gauge, Target } from "lucide-react";
import { api } from "@/lib/api";
import { StatCard } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { Skeleton } from "@/components/ui/skeleton";

interface StatsResponse {
  total_records?: number;
  fatal_rate?: number;
  avg_risk?: number;
  model_accuracy?: number;
  hourly_risk?: { hour: number; risk: number }[];
  city_risk?: { city: string; risk: number }[];
  weather_impact?: { weather: string; risk: number }[];
  city_ranking?: {
    city: string;
    total_accidents: number;
    fatal_accidents: number;
    avg_risk: number;
  }[];
}

const tooltipStyle = {
  contentStyle: {
    background: "hsl(var(--card))",
    border: "1px solid hsl(var(--border))",
    borderRadius: "0.75rem",
    color: "hsl(var(--foreground))",
    fontSize: "12px",
  },
};

const Dashboard = () => {
  const [data, setData] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
  // Fetch stats
  api
    .get("/stats")
    .then((res) => {
      const raw = res.data;

      // Convert dictionaries → arrays
      const hourly = Object.entries(raw.hourly_risk || {}).map(
        ([hour, risk]) => ({
          hour,
          risk,
        })
      );

      const city = Object.entries(raw.city_risk || {}).map(
        ([city, risk]) => ({
          city,
          risk,
        })
      );

      const weather = Object.entries(raw.weather_risk || {}).map(
        ([weather, risk]) => ({
          weather,
          risk,
        })
      );

      setData({
        total_records: raw.total_records,
        fatal_rate: raw.fatal_rate,
        avg_risk: raw.avg_risk,
        model_accuracy: raw.model_accuracy,
        hourly_risk: hourly,
        city_risk: city,
        weather_impact: weather,
      });
    })
    .catch((err) =>
      setError(err.message ?? "Failed to fetch stats")
    )
    .finally(() => setLoading(false));

  // Fetch city ranking separately
  api
    .get("/city-ranking")
    .then((res) => {
      setData((prev) => ({
        ...prev,
        city_ranking: res.data,
      }));
    })
    .catch((err) =>
      console.error("City ranking error:", err)
    );
  
  api.get("/model-performance").then((res) => {
    const models = res.data;

  // find best accuracy
    const best = Math.max(
      ...models.map((m: any) =>
        m.accuracy > 1
        ? m.accuracy
        : m.accuracy * 100
      )
    );

    setData((prev) => ({
      ...prev,
      model_accuracy: best,
    }));
});

}, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {error && (
        <div className="glass-panel border p-4 text-sm text-red-500">
          API unreachable — make sure backend is running.
        </div>
      )}

      {/* Stat Cards */}

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">

        <StatCard
          label="Total Records"
          value={data?.total_records?.toLocaleString() ?? "—"}
          icon={Database}
          accent="primary"
        />

        <StatCard
          label="Fatal Rate"
          value={
            data?.fatal_rate != null
              ? `${data.fatal_rate.toFixed(1)}%`
              : "—"
          }
          icon={AlertTriangle}
          accent="destructive"
        />

        <StatCard
          label="Avg Risk Score"
          value={
            data?.avg_risk != null
              ? data.avg_risk.toFixed(2)
              : "—"
          }
          icon={Gauge}
          accent="warning"
        />

        <StatCard
          label="Model Accuracy"
          value={
            data?.model_accuracy != null
              ? `${data.model_accuracy.toFixed(1)}%`
              : "—"
          }
          icon={Target}
          accent="success"
        />

      </div>

      {/* Hourly Risk Chart */}

      <ChartCard title="Hourly Risk Pattern">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data?.hourly_risk ?? []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip {...tooltipStyle} />

            <Line
              type="monotone"
              dataKey="risk"
              stroke="#3b82f6"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* City + Weather Charts */}

      <div className="grid gap-6 lg:grid-cols-2">

        <ChartCard title="City Risk Ranking">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data?.city_risk ?? []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="city" />
              <YAxis />
              <Tooltip {...tooltipStyle} />

              <Bar
                dataKey="risk"
                fill="#3b82f6"
              />

            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Weather Impact">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data?.weather_impact ?? []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="weather" />
              <YAxis />
              <Tooltip {...tooltipStyle} />

              <Bar
                dataKey="risk"
                fill="#60a5fa"
              />

            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

      </div>

      {/* City Table */}

      <ChartCard title="City Ranking Table">

        <div className="overflow-x-auto">

          <table className="w-full text-sm border-collapse">

            <thead>
              <tr className="border-b border-gray-700">

              <th className="px-4 py-2 text-left">#</th>

              <th className="px-4 py-2 text-left">
                City
              </th>

              <th className="px-4 py-2 text-center">
                Total
              </th>

              <th className="px-4 py-2 text-center">
                Fatal
              </th>

              <th className="px-4 py-2 text-center">
                Avg Risk
              </th>

            </tr>
          </thead>

        <tbody>

          {(data?.city_ranking ?? []).map(
            (row, i) => (
              <tr
                key={row.city}
                className="border-b border-gray-800 hover:bg-gray-800/40 transition"
              >

                <td className="px-4 py-2">
                  {i + 1}
                </td>

                <td className="px-4 py-2 font-medium">
                  {row.city}
                </td>

                <td className="px-4 py-2 text-center">
                  {row.total_accidents}
                </td>

                <td className="px-4 py-2 text-center">
                  {row.fatal_accidents}
                </td>

                <td className="px-4 py-2 text-center">
                  {row.avg_risk?.toFixed(2)}
                </td>

              </tr>
            )
          )}

        </tbody>

      </table>

        </div>

      </ChartCard>

    </div>
  );
};

export default Dashboard;