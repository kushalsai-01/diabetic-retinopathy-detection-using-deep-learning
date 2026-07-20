import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getHistory, HistoryItem, deletePrediction } from "../api";
import UrgencyBadge from "../components/UrgencyBadge";

export default function HistoryPage() {
  const navigate = useNavigate();

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const PAGE_SIZE = 10;

  const fetchHistory = () => {
    setLoading(true);
    setError(null);
    getHistory(page, PAGE_SIZE)
      .then((data) => {
        setItems(data.items);
        setTotal(data.total);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not load prediction history. Please verify database connection.");
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, [page]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this scan record? This cannot be undone.")) {
      return;
    }

    try {
      await deletePrediction(id);
      fetchHistory();
    } catch (err) {
      console.error(err);
      alert("Failed to delete record. Please try again.");
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE) || 1;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-teal-400">
            Diagnostic History Log
          </h1>
          <p className="text-sm text-gray-400">
            Review past scans, clinical triages, and print reports.
          </p>
        </div>
        <div className="text-xs font-mono text-gray-500 bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-800 self-start">
          Total Logs: {total}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-950/30 border border-red-900/50 rounded-xl text-sm text-red-400">
          ⚠️ {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[300px] gap-4">
          <div className="w-10 h-10 border-4 border-teal-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-400 text-sm">Fetching diagnostic history...</p>
        </div>
      ) : items.length === 0 ? (
        <div className="text-center p-12 bg-gray-900/40 border border-gray-800 rounded-2xl space-y-4">
          <div className="text-4xl text-gray-600">📁</div>
          <h3 className="text-md font-medium text-gray-300">No scans logged yet</h3>
          <p className="text-sm text-gray-500 max-w-sm mx-auto">
            Uploaded images and diagnosis records will appear here automatically.
          </p>
          <button
            onClick={() => navigate("/")}
            className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-gray-950 rounded-xl text-sm font-semibold transition-colors"
          >
            Start First Scan
          </button>
        </div>
      ) : (
        <div className="bg-gray-900/40 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-800 text-xs text-gray-400 font-semibold uppercase bg-gray-900/65">
                  <th className="px-6 py-4">Date</th>
                  <th className="px-6 py-4">Patient Info</th>
                  <th className="px-6 py-4">Filename</th>
                  <th className="px-6 py-4">Diagnosis</th>
                  <th className="px-6 py-4">Urgency</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 text-sm">
                {items.map((item) => (
                  <tr
                    key={item.prediction_id}
                    onClick={() => navigate(`/result/${item.prediction_id}`)}
                    className="hover:bg-gray-800/25 cursor-pointer transition-colors"
                  >
                    <td className="px-6 py-4 text-gray-400 text-xs font-mono">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 font-medium text-gray-200">
                      {item.patient_name || <span className="text-gray-600 text-xs italic">Anonymous</span>}
                    </td>
                    <td className="px-6 py-4 text-gray-400 truncate max-w-[150px]">
                      {item.image_filename}
                    </td>
                    <td className="px-6 py-4">
                      {item.grade !== null ? (
                        <span className="font-semibold text-teal-400">
                          {item.grade_name} (G{item.grade})
                        </span>
                      ) : (
                        <span className="text-red-400 font-medium">Quality Failed</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      {item.urgency ? (
                        <UrgencyBadge urgency={item.urgency as any} />
                      ) : (
                        <span className="text-gray-500">—</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={(e) => handleDelete(item.prediction_id, e)}
                        className="p-2 text-red-400 hover:text-red-300 hover:bg-red-950/20 rounded-lg transition-colors"
                        title="Delete record"
                      >
                        🗑️
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          <div className="flex items-center justify-between border-t border-gray-800 px-6 py-4 bg-gray-900/65">
            <button
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              disabled={page === 1}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-800 ${
                page === 1
                  ? "text-gray-600 cursor-not-allowed"
                  : "text-gray-400 hover:text-gray-200 bg-gray-850"
              }`}
            >
              Previous
            </button>
            <span className="text-xs text-gray-400">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
              disabled={page === totalPages}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-800 ${
                page === totalPages
                  ? "text-gray-600 cursor-not-allowed"
                  : "text-gray-400 hover:text-gray-200 bg-gray-850"
              }`}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
