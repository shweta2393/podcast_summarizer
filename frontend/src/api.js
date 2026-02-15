import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
});

export async function uploadAudio(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function submitUrl(url) {
  const response = await api.post("/api/url", { url });
  return response.data;
}

export async function pollJobStatus(jobId) {
  const response = await api.get(`/api/status/${jobId}`);
  return response.data;
}
