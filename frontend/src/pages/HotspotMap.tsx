import { ExternalLink, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { API_BASE } from "@/lib/api";

const HotspotMap = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight flex items-center gap-3">
            <MapPin className="h-7 w-7 text-primary" />
            Hotspot Map
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Interactive geographic visualization of accident hotspots across India.
          </p>
        </div>
        <Button variant="outline" asChild>
          <a href={`${API_BASE}/map`} target="_blank" rel="noreferrer">
            <ExternalLink className="mr-2 h-4 w-4" />
            Open in new tab
          </a>
        </Button>
      </div>

      <div className="glass-panel overflow-hidden p-2">
        <iframe
          src={`${API_BASE}/map`}
          title="Accident hotspot map"
          className="h-[78vh] w-full rounded-lg border border-border bg-secondary/30"
        />
      </div>
    </div>
  );
};

export default HotspotMap;
