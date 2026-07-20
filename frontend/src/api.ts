import axios from "axios";

// Use VITE_API_URL env var in production, fallback to localhost in dev
const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

// ---- Types ----------------------------------------------------------------

export interface PredictionResponse {
  prediction_id: string;
  grade: number;
  grade_name: string;
  probabilities: number[];
  recommendation: string;
  urgency: "routine" | "soon" | "urgent" | "emergency";
  heatmap_url: string | null;
  pdf_url: string | null;
  image_url: string | null;
  quality_passed: boolean;
  quality_reason: string | null;
  created_at: string;
}

export interface HistoryItem {
  prediction_id: string;
  image_filename: string;
  patient_name: string | null;
  grade: number | null;
  grade_name: string | null;
  urgency: string | null;
  created_at: string;
}

export interface PaginatedHistory {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

// ---- API calls ------------------------------------------------------------

/**
 * Sends image and optional patient metadata for DR prediction.
 * Returns the full prediction result.
 */
export async function predictImage(
  file: File,
  patientName?: string,
  patientDob?: string,
  patientId?: string
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  if (patientName) form.append("patient_name", patientName);
  if (patientDob) form.append("patient_dob", patientDob);
  if (patientId) form.append("patient_id", patientId);
  const { data } = await api.post<PredictionResponse>("/api/predict", form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return data;
}

/**
 * Fetches paginated prediction history.
 */
export async function getHistory(
  page: number = 1,
  pageSize: number = 10
): Promise<PaginatedHistory> {
  const { data } = await api.get<PaginatedHistory>("/api/history", {
    params: { page, page_size: pageSize },
  });
  return data;
}

/**
 * Fetches a single prediction by ID.
 */
export async function getPrediction(id: string): Promise<PredictionResponse> {
  const { data } = await api.get<PredictionResponse>(`/api/history/${id}`);
  return data;
}

/**
 * Deletes a prediction by ID.
 */
export async function deletePrediction(id: string): Promise<void> {
  await api.delete(`/api/history/${id}`);
}
export default api;
