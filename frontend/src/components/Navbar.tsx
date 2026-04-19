import { NavLink, useLocation } from "react-router-dom";
import { Activity, BarChart3, Bot, Map, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { to: "/", label: "Dashboard", icon: BarChart3 },
  { to: "/predict", label: "Predict Risk", icon: Sparkles },
  { to: "/map", label: "Hotspot Map", icon: Map },
  { to: "/performance", label: "Model Performance", icon: Activity },
  { to: "/assistant", label: "AI Assistant", icon: Bot },
];

export const Navbar = () => {
  const location = useLocation();
  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <div className="container flex h-16 items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-primary shadow-[0_0_20px_hsl(var(--primary)/0.5)]">
            <Activity className="h-5 w-5 text-primary-foreground" strokeWidth={2.5} />
          </div>
          <div className="hidden sm:block">
            <div className="font-display text-sm font-semibold leading-tight tracking-tight">Indian Road Accident</div>
            <div className="font-display text-xs text-muted-foreground leading-tight">Risk Analyzer</div>
          </div>
        </div>

        <nav className="flex items-center gap-1 overflow-x-auto">
          {links.map(({ to, label, icon: Icon }) => {
            const active = location.pathname === to;
            return (
              <NavLink
                key={to}
                to={to}
                className={cn(
                  "flex items-center gap-2 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium transition-all",
                  active
                    ? "bg-primary/15 text-primary shadow-[0_0_20px_hsl(var(--primary)/0.15)]"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden md:inline">{label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
