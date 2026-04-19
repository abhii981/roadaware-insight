import { useState } from "react";
import { Loader2, Sparkles, ShieldAlert, ShieldCheck, Shield } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { ChartCard } from "@/components/ChartCard";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

const cities = ["Mumbai", "Delhi", "Pune", "Chennai", "Bangalore", "Hyderabad", "Kolkata", "Chandigarh"];
const weathers = ["Clear", "Rain", "Fog"];
const visibilities = ["Low", "Medium", "High"];
const roadTypes = ["Highway", "Urban", "Rural"];
const trafficDensities = ["Low", "Medium", "High"];
const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const festivals = ["None", "Diwali", "Holi", "Eid", "New Year"];
const models = ["Random Forest", "XGBoost", "KNN", "Logistic Regression"];

interface PredictionResult {
  risk_level?: "LOW" | "MEDIUM" | "HIGH" | string;
  probabilities?: { LOW?: number; MEDIUM?: number; HIGH?: number; [k: string]: number | undefined };
  confidence?: number;
  message?: string;
}

const SelectField = ({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) => (
  <div className="space-y-2">
    <Label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</Label>
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="bg-input/50">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o} value={o}>{o}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  </div>
);

const SliderField = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}) => (
  <div className="space-y-3">
    <div className="flex items-center justify-between">
      <Label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</Label>
      <span className="rounded-md bg-primary/10 px-2 py-0.5 font-mono text-xs text-primary">
        {step < 1 ? value.toFixed(2) : value}
      </span>
    </div>
    <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
  </div>
);

const PredictRisk = () => {
  const [form, setForm] = useState({
    city: "Mumbai",
    weather: "Clear",
    visibility: "High",
    road_type: "Highway",
    traffic_density: "Medium",
    day_of_week: "Monday",
    festival: "None",
    primary_cause: "Speeding",
    hour: 12,
    temperature: 28,
    lanes: 2,
    vehicles_involved: 2,
    casualties: 1,
    risk_score: 0.5,
    weekend: false,
    peak_hour: false,
    traffic_signal: true,
    model: "Random Forest",
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const update = <K extends keyof typeof form>(key: K, value: typeof form[K]) =>
    setForm((p) => ({ ...p, [key]: value }));

  const handleSubmit = async () => {
  setLoading(true);
  setResult(null);

  try {
    // ONLY mapping names — UI stays untouched
    const payload = {
      city: form.city,
      weather: form.weather,
      visibility: form.visibility,
      road_type: form.road_type,
      traffic_density: form.traffic_density,
      day_of_week: form.day_of_week,
      festival: form.festival,

      cause: form.primary_cause,

      hour: Number(form.hour),
      temperature: Number(form.temperature),
      lanes: Number(form.lanes),
      vehicles_involved: Number(form.vehicles_involved),
      casualties: Number(form.casualties),
      risk_score: Number(form.risk_score),

      is_weekend: Boolean(form.weekend),
      is_peak_hour: Boolean(form.peak_hour),
      traffic_signal: Boolean(form.traffic_signal)
    };

    console.log("Payload:", payload);

    const res = await api.post<PredictionResult>(
      "/predict",
      payload
    );

    setResult(res.data);

  } catch (e: any) {
    console.error(e);

    toast.error(
      e?.response?.data?.detail ||
      "Prediction failed"
    );

  } finally {
    setLoading(false);
  }
};

  const riskMeta = (lvl?: string) => {
    if (lvl === "HIGH") return { color: "text-destructive", bg: "bg-destructive/10", border: "border-destructive/40", icon: ShieldAlert };
    if (lvl === "MEDIUM") return { color: "text-warning", bg: "bg-warning/10", border: "border-warning/40", icon: Shield };
    return { color: "text-success", bg: "bg-success/10", border: "border-success/40", icon: ShieldCheck };
  };

  const probColors: Record<string, string> = {
    LOW: "bg-success",
    MEDIUM: "bg-warning",
    HIGH: "bg-destructive",
  };

  const meta = riskMeta(result?.risk_level);
  const Icon = meta.icon;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">Predict Risk</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure scenario parameters and run an AI model to assess accident risk.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <ChartCard title="Scenario Parameters" description="Categorical inputs">
            <div className="grid gap-4 md:grid-cols-2">
              <SelectField label="City" value={form.city} onChange={(v) => update("city", v)} options={cities} />
              <SelectField label="Weather" value={form.weather} onChange={(v) => update("weather", v)} options={weathers} />
              <SelectField label="Visibility" value={form.visibility} onChange={(v) => update("visibility", v)} options={visibilities} />
              <SelectField label="Road Type" value={form.road_type} onChange={(v) => update("road_type", v)} options={roadTypes} />
              <SelectField label="Traffic Density" value={form.traffic_density} onChange={(v) => update("traffic_density", v)} options={trafficDensities} />
              <SelectField label="Day of Week" value={form.day_of_week} onChange={(v) => update("day_of_week", v)} options={days} />
              <SelectField label="Festival" value={form.festival} onChange={(v) => update("festival", v)} options={festivals} />
              <div className="space-y-2">
                <Label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Primary Cause</Label>
                <Input
                  value={form.primary_cause}
                  onChange={(e) => update("primary_cause", e.target.value)}
                  className="bg-input/50"
                  placeholder="e.g. Speeding"
                />
              </div>
            </div>
          </ChartCard>

          <ChartCard title="Numerical Parameters" description="Adjust the sliders">
            <div className="grid gap-6 md:grid-cols-2">
              <SliderField label="Hour of Day" value={form.hour} onChange={(v) => update("hour", v)} min={0} max={23} />
              <SliderField label="Temperature (°C)" value={form.temperature} onChange={(v) => update("temperature", v)} min={5} max={50} />
              <SliderField label="Lanes" value={form.lanes} onChange={(v) => update("lanes", v)} min={1} max={6} />
              <SliderField label="Vehicles Involved" value={form.vehicles_involved} onChange={(v) => update("vehicles_involved", v)} min={1} max={10} />
              <SliderField label="Casualties" value={form.casualties} onChange={(v) => update("casualties", v)} min={0} max={10} />
              <SliderField label="Risk Score" value={form.risk_score} onChange={(v) => update("risk_score", v)} min={0} max={1} step={0.01} />
            </div>
          </ChartCard>

          <ChartCard title="Conditions & Model" description="Boolean flags and model selection">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3">
                {[
                  { key: "weekend", label: "Weekend" },
                  { key: "peak_hour", label: "Peak Hour" },
                  { key: "traffic_signal", label: "Traffic Signal Present" },
                ].map(({ key, label }) => (
                  <label key={key} className="flex items-center gap-3 rounded-lg border border-border bg-secondary/30 p-3 transition-colors hover:bg-secondary/60 cursor-pointer">
                    <Checkbox
                      checked={form[key as keyof typeof form] as boolean}
                      onCheckedChange={(c) => update(key as any, !!c)}
                    />
                    <span className="text-sm font-medium">{label}</span>
                  </label>
                ))}
              </div>

              <div>
                <Label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Model</Label>
                <RadioGroup value={form.model} onValueChange={(v) => update("model", v)} className="mt-3 space-y-2">
                  {models.map((m) => (
                    <label
                      key={m}
                      className={cn(
                        "flex items-center gap-3 rounded-lg border p-3 cursor-pointer transition-all",
                        form.model === m
                          ? "border-primary/60 bg-primary/10"
                          : "border-border bg-secondary/30 hover:bg-secondary/60"
                      )}
                    >
                      <RadioGroupItem value={m} />
                      <span className="text-sm font-medium">{m}</span>
                    </label>
                  ))}
                </RadioGroup>
              </div>
            </div>
          </ChartCard>

          <Button
            onClick={handleSubmit}
            disabled={loading}
            size="lg"
            className="w-full bg-gradient-primary text-primary-foreground font-semibold text-base shadow-[0_0_30px_hsl(var(--primary)/0.4)] hover:shadow-[0_0_40px_hsl(var(--primary)/0.6)] transition-all"
          >
            {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Sparkles className="mr-2 h-5 w-5" />}
            Analyze Risk
          </Button>
        </div>

        <div className="lg:col-span-1">
          <div className="sticky top-24">
            <ChartCard title="Prediction Result" description="Live model output">
              {!result && !loading && (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-secondary/50">
                    <Sparkles className="h-7 w-7 text-muted-foreground" />
                  </div>
                  <p className="mt-4 text-sm text-muted-foreground">
                    Submit the form to see the AI risk assessment
                  </p>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center py-16">
                  <Loader2 className="h-10 w-10 animate-spin text-primary" />
                  <p className="mt-4 text-sm text-muted-foreground">Running model...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6 animate-fade-in">
                  <div className={cn("rounded-xl border p-5", meta.bg, meta.border)}>
                    <div className="flex items-center gap-3">
                      <Icon className={cn("h-8 w-8", meta.color)} />
                      <div>
                        <p className="text-xs uppercase tracking-wider text-muted-foreground">Risk Level</p>
                        <p className={cn("font-display text-2xl font-bold", meta.color)}>
                          {result.risk_level ?? "—"}
                        </p>
                      </div>
                    </div>
                  </div>

                  {result.probabilities && (
                    <div className="space-y-3">
                      <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Class Probabilities
                      </p>
                      {Object.entries(result.probabilities).map(([cls, p]) => {
                        const pct = Math.round((p ?? 0) * 100);
                        return (
                          <div key={cls}>
                            <div className="mb-1 flex items-center justify-between text-xs">
                              <span className="font-medium">{cls}</span>
                              <span className="font-mono text-muted-foreground">{pct}%</span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-secondary">
                              <div
                                className={cn("h-full rounded-full transition-all", probColors[cls] ?? "bg-primary")}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {result.confidence != null && (
                    <div className="rounded-lg border border-border bg-secondary/30 p-3">
                      <p className="text-xs text-muted-foreground">Confidence</p>
                      <p className="font-mono text-lg font-semibold">{(result.confidence * 100).toFixed(1)}%</p>
                    </div>
                  )}

                  {result.message && <p className="text-sm text-muted-foreground">{result.message}</p>}
                </div>
              )}
            </ChartCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictRisk;
