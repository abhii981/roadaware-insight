import axios from "axios";

export const API_BASE = "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});
