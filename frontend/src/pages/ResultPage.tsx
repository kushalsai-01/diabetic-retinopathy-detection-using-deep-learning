import { useParams, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { getPrediction, PredictionResponse } from "../api";
import GradeBar from "../components/GradeBar";
import UrgencyBadge from "../components/UrgencyBadge";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function ResultPage() {
  const { predictionId } = useParams<{ predictionId: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If we came from the upload redirect, we have result in location state
    if (location.state?.result) {
      setResult(location.state.result);
      return;
    }

    // Otherwise, fetch it by ID (direct URL access/refresh)
    if (predictionId) {
      setLoading(true);
      setError(null);
      getPrediction(predictionId)
        .then((data) => {
          setResult(data);
        })
        .catch((err) => {
          console.error(err);
          setError("Failed to load diagnostic results. Please check prediction ID.");
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [predictionId, location.state]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400">Loading diagnostic report...</p>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-md mx-auto text-center space-y-4 p-8 bg-gray-900/60 border border-gray-800 rounded-2xl">
        <div className="text-red-400 text-3xl">⚠️</div>
        <h2 className="text-lg font-bold text-gray-200">Error Loading Results</h2>
        <p className="text-sm text-gray-400">{error || "Diagnostic report not found."}</p>
        <button
          onClick={() => navigate("/")}
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-xl text-sm font-medium transition-colors"
        >
          Return to Upload
        </button>
      </div>
    );
  }

  // Construct absolute image/PDF URLs
  const originalImageUrl = result.image_url ? `${BASE_URL}${result.image_url}` : null;
  const heatmapUrl = result.heatmap_url ? `${BASE_URL}${result.heatmap_url}` : null;
  const pdfUrl = result.pdf_url ? `${BASE_URL}${result.pdf_url}` : null;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-6 bg-gray-900/40 rounded-2xl border border-gray-800">
        <div className="space-y-1">
          <div className="text-xs text-gray-500 font-mono uppercase">Prediction ID: {result.prediction_id}</div>
          <h2 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
            Status: <span className="text-teal-400">{result.grade_name}</span>
          </h2>
          <p className="text-xs text-gray-400">
            Exam Date: {new Date(result.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-gray-400 uppercase">Triage:</span>
          <UrgencyBadge urgency={result.urgency} />
        </div>
      </div>

      {/* Main Grid: Images + Report Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Images comparisons */}
        <div className="space-y-6">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
              Diagnostic Visualization
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {/* Original image */}
              <div className="space-y-1 text-center">
                <div className="bg-gray-900 rounded-xl overflow-hidden aspect-square border border-gray-800 flex items-center justify-center">
                  {originalImageUrl ? (
                    <img
                      src={originalImageUrl}
                      alt="Original fundus scan"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-gray-600">No original image</span>
                  )}
                </div>
                <span className="text-[11px] text-gray-500 font-medium">Original Fundus Image</span>
              </div>

              {/* Grad-CAM */}
              <div className="space-y-1 text-center">
                <div className="bg-gray-900 rounded-xl overflow-hidden aspect-square border border-gray-800 flex items-center justify-center">
                  {heatmapUrl ? (
                    <img
                      src={heatmapUrl}
                      alt="GradCAM heatmap overlay"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-gray-600">No Grad-CAM overlay</span>
                  )}
                </div>
                <span className="text-[11px] text-teal-500 font-medium">Grad-CAM Heatmap</span>
              </div>
            </div>
          </div>

          {/* Clinical Recommendation Text Box */}
          <div className="p-6 bg-teal-950/20 border border-teal-900/50 rounded-2xl space-y-2">
            <h4 className="text-sm font-semibold text-teal-400">Clinical Guidelines</h4>
            <p className="text-sm text-gray-300 leading-relaxed">{result.recommendation}</p>
          </div>
        </div>

        {/* Probabilities / Report Actions */}
        <div className="space-y-6">
          <GradeBar probabilities={result.probabilities} predictedGrade={result.grade} />

          {/* Report Downloads and Navigation */}
          <div className="flex flex-col sm:flex-row gap-4">
            {pdfUrl && (
              <a
                href={pdfUrl}
                target="_blank"
                rel="noreferrer"
                className="flex-1 py-3 bg-teal-500 hover:bg-teal-400 text-gray-950 rounded-xl font-semibold text-center transition-all shadow-lg shadow-teal-500/20 flex items-center justify-center gap-2"
              >
                📄 Download PDF Report
              </a>
            )}
            <button
              onClick={() => navigate("/")}
              className="flex-1 py-3 bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700 rounded-xl font-semibold transition-all"
            >
              Analyze Another Scan
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
