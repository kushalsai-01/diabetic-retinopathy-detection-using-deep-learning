import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { predictImage } from "../api";

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  const [patientName, setPatientName] = useState("");
  const [patientDob, setPatientDob] = useState("");
  const [patientId, setPatientId] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (PNG, JPG).");
      return;
    }
    setError(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const removeImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please select a retinal fundus image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictImage(
        selectedFile,
        patientName || undefined,
        patientDob || undefined,
        patientId || undefined
      );
      
      // Navigate to results page with prediction_id and result data
      navigate(`/result/${result.prediction_id}`, { state: { result } });
    } catch (err: any) {
      console.error(err);
      setError(
        err.response?.data?.detail || 
        "Analysis failed. Please verify the backend is online and the image is not corrupted."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-teal-400">
          Retinal Fundus Image Analysis
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto">
          Upload a high-resolution fundus photograph to perform real-time screening for Diabetic Retinopathy.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload Zone */}
        <div className="space-y-4">
          <label className="block text-sm font-semibold text-gray-300">
            Select Fundus Image
          </label>
          <div
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            className={`relative min-h-[300px] border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-6 transition-all duration-200 ${
              dragActive 
                ? "border-teal-500 bg-teal-500/5 scale-[0.99]" 
                : "border-gray-800 bg-gray-900/40 hover:border-gray-700 hover:bg-gray-900/60"
            }`}
          >
            {previewUrl ? (
              <div className="w-full flex flex-col items-center gap-4">
                <img
                  src={previewUrl}
                  alt="Fundus preview"
                  className="max-h-60 rounded-xl object-contain border border-gray-800"
                />
                <button
                  type="button"
                  onClick={removeImage}
                  className="px-3 py-1.5 text-xs font-medium text-red-400 bg-red-950/40 border border-red-900/50 rounded-lg hover:bg-red-900/40 hover:text-red-300 transition-colors"
                >
                  Remove Image
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center text-center gap-4">
                <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center text-gray-400">
                  📷
                </div>
                <div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-teal-400 hover:text-teal-300 font-semibold focus:outline-none"
                  >
                    Click to upload
                  </button>
                  <span className="text-gray-400"> or drag and drop</span>
                  <p className="text-xs text-gray-500 mt-2">
                    Supports PNG, JPG, JPEG (Max 15MB)
                  </p>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        </div>

        {/* Patient Metadata Form */}
        <form onSubmit={handleSubmit} className="space-y-6 flex flex-col justify-between">
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
              Patient Information (Optional)
            </h3>
            
            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-400">
                Patient ID / MRN
              </label>
              <input
                type="text"
                placeholder="e.g. MRN-10294"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-all"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-400">
                Patient Name
              </label>
              <input
                type="text"
                placeholder="e.g. John Doe"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-all"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-400">
                Date of Birth
              </label>
              <input
                type="date"
                value={patientDob}
                onChange={(e) => setPatientDob(e.target.value)}
                className="w-full bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-all [color-scheme:dark]"
              />
            </div>
          </div>

          <div className="space-y-4 pt-4">
            {error && (
              <div className="p-4 bg-red-950/30 border border-red-900/50 rounded-xl text-sm text-red-400">
                ⚠️ {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !selectedFile}
              className={`w-full py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
                loading
                  ? "bg-teal-900/40 text-teal-400 border border-teal-900 cursor-not-allowed"
                  : selectedFile
                  ? "bg-teal-500 hover:bg-teal-400 text-gray-950 shadow-lg shadow-teal-500/20 active:scale-[0.98]"
                  : "bg-gray-800 text-gray-500 border border-gray-900 cursor-not-allowed"
              }`}
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-teal-400 border-t-transparent rounded-full animate-spin" />
                  Running Neural Analysis...
                </>
              ) : (
                "Run Diagnostic Scan"
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
