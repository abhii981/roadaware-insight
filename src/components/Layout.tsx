import { Outlet } from "react-router-dom";
import { Navbar } from "./Navbar";

export const Layout = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="container py-8 animate-fade-in">
        <Outlet />
      </main>
    </div>
  );
};
